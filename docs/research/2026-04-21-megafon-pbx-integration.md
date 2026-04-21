# Интеграция голосового AI-ассистента с виртуальной АТС МегаФон

*Research note, 2026-04-21. Проект: КЭСИ voice assistant (`backend/main.py`, WebSocket + VAD + Whisper + Qwen + Silero TTS).*

## TL;DR

- МегаФон сам по себе **не предоставляет** программируемого медиа-API уровня Twilio/Voximplant: его Виртуальная АТС — это классическая облачная PBX с REST-интеграциями для CRM и webhook-ами о событиях звонка, но **без сырого аудиопотока** в приложение.
- Единственный реалистичный путь «через МегаФон» — подключить его номер **SIP-транком к собственному Asterisk/FreeSWITCH**, а уже оттуда мостить медиа в наш Python-бэкенд через `ExternalMedia`/`AudioSocket` и управлять переводом из ARI.
- «Сценарий возврата на ассистента при no-answer» **не реализуется нативно ни в МегаФон ВАТС, ни в blind-transfer SIP-мире** — его придётся собирать руками в dialplan/ARI (таймер + `Dial` с failover-ом на обратный канал к ассистенту).
- **Exolve — это МТС, не МегаФон** (типичное заблуждение). Из российских CPaaS реально под задачу подходят: MTS Exolve Voice API, Voximplant (теперь Kaspersky), UIS/Comagic, Mango Office.
- Рекомендация: для MVP — Asterisk + SIP-транк МегаФона; для прод-сценария с минимумом телеком-возни — MTS Exolve или Voximplant.

---

## 1. Продукты МегаФон для бизнеса

| Продукт | Что это | API для нашего сценария |
|---|---|---|
| **Виртуальная АТС** (`vats.megafon.ru`) | Облачная PBX: многоканальные номера, IVR, очереди, запись. | REST API для CRM-интеграции + webhook о событиях звонка (ringing/answered/hangup). **Нет** программного доступа к медиа-потоку и нет программируемого dialplan-а. |
| **МультиФон Бизнес** | SIP-телефония, по сути SIP-транк провайдера. | Стандартный SIP — подходит как PSTN-шлюз перед своим Asterisk. |
| **Голосовой робот МегаФон** | Готовый SaaS робот-обзвонщик с интеграцией по API/SMPP. | Закрытое решение; свой LLM туда не подключишь. |
| **«МегаФон Exolve»** | **Не существует.** Exolve — продукт МТС (`exolve.ru`, `docs.exolve.ru`). |  |
| **Подключение внешних SIP-АТС к ВАТС** | Режим, в котором собственный Asterisk входит в номерной план ВАТС по SIP-регистрации или static-IP транку (`sbc.megafon.ru`). Требует тарифа «Большой Бизнес». | **Ключевая точка входа** для нашего сценария. |

Публичной developer-документации уровня Twilio/Exolve у МегаФона **нет**: REST-спецификация ВАТС скачивается из личного кабинета, webhook-события ограничены CRM-нуждами (звонок пришёл / завершился / запись готова), программно инициировать перевод на очередь через REST — возможно, но возврат потока в приложение через API не предусмотрен.

## 2. Warm transfer + no-answer fallback

Целевой сценарий (перевод → ждём 20 с → если не сняли, вернулось к ассистенту) **нативно не поддерживается**:

- В Виртуальной АТС МегаФона доступен *attended transfer* как DTMF/операторское действие, но логика «вернуть звонок инициатору при no-answer» — это внутренняя фишка PBX, и для робота-инициатора её поведение не документировано и не программируется.
- В SIP/Asterisk мире blind transfer (REFER) по умолчанию **не возвращается** — если удалённый не ответил, вызов просто падает. Чтобы получить «return on no-answer», надо:
  1. Не делать REFER наружу, а держать звонок под контролем своего Asterisk.
  2. Вызывать оператора через `Dial(..., T, g)` с таймаутом, и в dialplan после возврата `Dial` (статус `NOANSWER`/`CHANUNAVAIL`) **переподключать** канал обратно к Stasis-приложению ассистента.
  3. Либо через ARI: положить вызывающего в `holding bridge`, запустить `originate` на оператора; при `StasisEnd`/`Dial failure` — вернуть канал в bridge с ассистентом и воспроизвести реплику «операторы заняты».
- В Asterisk 20+ для этого есть `channelTransferEvent` (`PJSIP_TRANSFER_HANDLING=ari-only`), позволяющий перехватывать и отменять transfer-события из приложения.

Вывод: «умный» возврат — это **наша ответственность на уровне Asterisk ARI**, а не функция оператора.

## 3. Архитектура подключения WebSocket-бэкенда к SIP-миру

Прямо подключить наш `/ws` эндпоинт к телефонной сети **нельзя** — между PSTN и Python всегда живёт SIP-софтсвитч. Типовая схема:

```
PSTN (номер МегаФона)
   │  SIP (SBC МегаФона: sbc.megafon.ru)
   ▼
[ Asterisk / FreeSWITCH  ]  ← регистрируется как внешняя АТС у ВАТС
   │                         или получает SIP-транк МультиФона
   │  ARI (REST/WebSocket control plane)
   │  ExternalMedia / AudioSocket (RTP↔TCP audio)
   ▼
[ Python bridge ]  — конвертирует G.711 µ-law ↔ 16 kHz PCM,
   │                  проксирует аудио в существующий /ws
   ▼
[ FastAPI backend (main.py) ]  — VAD + Whisper + Qwen + Silero TTS
```

Практические варианты моста «Asterisk ↔ Python»:

- **AudioSocket** (`app_audiosocket`, Asterisk 18+) — TCP, очень простой протокол, идеален для «один звонок — один сокет». Отлично ложится на наш текущий WebSocket-интерфейс: достаточно тонкого адаптера.
- **ARI `externalMedia`** — RTP-поток уходит на указанный host:port, управление каналом — через ARI WebSocket. Более гибко (barge-in, перевод, hold), но сложнее в реализации.
- **MRCP / NGS / Voximplant Kit** — готовые «коробки», но требуют переписать пайплайн под их SDK.

Готовые опенсорс-референсы, которые можно форкнуть: `hkjarral/Asterisk-AI-Voice-Agent`, `vaheed/sip-ai-agent`, `aicc2025/sip-to-ai` — все реализуют SIP↔OpenAI Realtime по схеме, идентичной нашей (заменить OpenAI на `main.py`).

## 4. Сложность и стоимость

- **Получение номера и SIP-транка у МегаФона.** 1–3 недели: договор с юрлицом, включение опции «Большой Бизнес» на ВАТС, предоставление статического IP для SBC, настройка кодеков (G.711a/µ, opus опционально). Тестовой песочницы нет — сразу боевой транк, но с ограниченным лимитом.
- **Своя Asterisk-инсталляция** с `app_audiosocket` + Python-bridge: 1–2 недели инженерной работы, если брать готовый референс из п. 3.
- **Логика warm-transfer + return-on-no-answer в ARI:** ещё 1 неделя включая тестирование edge-cases (оператор взял трубку и сбросил; no-answer таймер; абонент положил трубку во время звонка оператору).
- **Типичные подводные камни:**
  - Whisper плохо жуёт 8 kHz G.711 — обязательно апсемплить до 16 kHz перед STT.
  - SBC МегаФона капризен к NAT; обычно нужен либо белый IP, либо TLS+SRTP поверх VPN.
  - Нет тестовой среды — отладка переводов ведётся на реальных минутах (платных).
  - Запись разговора должна соответствовать 152-ФЗ: либо храним у себя и удаляем по запросу, либо пишем на стороне ВАТС.

## 5. Альтернативы в рамках РФ

| Платформа | Плюсы | Минусы |
|---|---|---|
| **MTS Exolve** (`exolve.ru`) | Программируемый Voice API, SIP-trunk API, Numbering API, документация и Postman-коллекции публичны, есть GitHub-примеры, можно поднимать SIP-приложения и управлять маршрутизацией вызовов через REST. | Своего «bring-your-own-LLM» media-streaming API нет в явном виде — часть сценариев только через их готового робота. Warm-transfer с возвратом всё равно собирается на своём SIP-узле. |
| **Voximplant** (Kaspersky) | Полноценный CPaaS: VoxEngine-скрипты (JS-dialplan), WebSocket media streaming, Dialogflow/Tinkoff VoiceKit, Russian TTS/STT, SIP + WebRTC, `call.transfer()` с fallback-колбэком. Ближе всего к «Twilio для РФ». | Платформа проприетарная; vendor lock-in; нумерация РФ через их партнёров. |
| **UIS / Comagic** | Сильная ВАТС + API + webhook-и событий, популярен для контакт-центров. | Медиа-API ограниченный — больше про маршрутизацию и аналитику, чем про AI-voice. |
| **Mango Office** | Готовый голосовой робот + REST API ВАТС, гибридный режим робот↔оператор. | Закрытая экосистема, свой LLM не подключить; годится только если «ассистент» = их робот. |

## Рекомендация для нашего проекта

**Двухшаговый план:**

1. **MVP (2–4 недели):** арендуем один номер у МегаФона как SIP-транк (МультиФон Бизнес), поднимаем Asterisk 20 в том же docker-compose, используем `AudioSocket` → адаптер → существующий `/ws` в `backend/main.py`. Логика перевода:
   - ARI-приложение держит абонента в `Stasis`-сессии с ассистентом.
   - По tool-call `transfer_to_operator` запускаем `Dial(PJSIP/queue,20,g)` параллельно с holding-тоном.
   - Если `DIALSTATUS` != `ANSWER` → возвращаем канал в Stasis, ассистент говорит «операторы заняты» и вызывает существующий `create_ticket` в Bitrix24.
   - Если `ANSWER` → `Bridge` абонента с оператором, ассистент отключается.
2. **Прод (если минуты МегаФона окажутся дороже разработки):** мигрировать SIP-транк на **MTS Exolve** — их Voice/SIP API даёт более предсказуемый биллинг и нормальный dev-портал, при этом Asterisk-логика из MVP остаётся без изменений.

**Не стоит пытаться** реализовать сценарий напрямую через REST API Виртуальной АТС МегаФона — он не рассчитан на программируемый медиа-сценарий, и «возврат при no-answer» там не собирается.

---

### Ссылки

- Виртуальная АТС, REST API: <https://vats.megafon.ru/rest_api>
- Подключение внешних SIP-АТС к ВАТС МегаФон: <https://vats.megafon.ru/sippbx>
- МультиФон Бизнес (SIP для бизнеса): <https://multifon.megafon.ru/>
- Голосовой робот МегаФон: <https://nn.megafon.ru/corporate/services/golosovoy_robot>
- MTS Exolve Voice API: <https://exolve.ru/products/voice-api/> и <https://docs.exolve.ru/docs/ru/api-reference/voice-api/>
- MTS Exolve SIP API: <https://exolve.ru/products/sip/>
- MTS Exolve GitHub (образцы): <https://github.com/mtsexolve>
- Voximplant AI/NLP integrations: <https://voximplant.com/platform/ai-and-nlp>
- Asterisk ARI Transfer Handling: <https://docs.asterisk.org/Configuration/Interfaces/Asterisk-REST-Interface-ARI/Introduction-to-ARI-Transfer-Handling/>
- AudioSocket + ARI ExternalMedia дискуссия: <https://community.asterisk.org/t/audiosocket-and-ari-externalmedia/112258>
- Референс-проекты SIP↔AI (форкать можно):
  - <https://github.com/hkjarral/Asterisk-AI-Voice-Agent>
  - <https://github.com/vaheed/sip-ai-agent>
  - <https://github.com/aicc2025/sip-to-ai>
- Tutorial «AI voice agent + Asterisk SIP + Python» (2025): <https://towardsai.net/p/machine-learning/how-to-build-an-ai-voice-agent-with-openai-realtime-api-asterisk-sip-2025-using-python-with-github-repo>
- Mango Office API: <https://www.mango-office.ru/support/api/>
