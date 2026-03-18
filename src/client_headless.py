import asyncio
import websockets
import wave
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def send_audio(websocket, file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            if wf.getframerate() != 16000 or wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                logger.warning("Input file should be 16kHz, 16-bit, Mono. Found incorrect format!")
            
            # Read and send 1024 bytes (512 frames of 16-bit audio) chunks
            data = wf.readframes(512)
            chunk_count = 0
            while data:
                await websocket.send(data)
                await asyncio.sleep(0.032)
                data = wf.readframes(512)
                chunk_count += 1
                
        logger.info(f"Finished sending {chunk_count} audio chunks.")
        
        logger.info("Sending 100 chunks of silence to trigger VAD endpointing...")
        silence_chunk = b'\x00' * 1024
        for _ in range(100):
            await websocket.send(silence_chunk)
            await asyncio.sleep(0.032)
            
        logger.info("Finished sending silence.")
    except Exception as e:
        logger.error(f"Error in send_audio: {e}")

async def receive_messages(websocket):
    audio_data = bytearray()
    
    try:
        while True:
            try:
                # 15 seconds timeout for receiving a message. 
                # If server takes longer to process, you can increase this value.
                message = await asyncio.wait_for(websocket.recv(), timeout=15.0)
            except asyncio.TimeoutError:
                logger.info("No messages received for 15 seconds. Assuming server finished sending response.")
                break
                
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    logger.info("Received JSON:")
                    print(json.dumps(data, indent=4, ensure_ascii=False))
                except json.JSONDecodeError:
                    logger.info(f"Received text: {message}")
            elif isinstance(message, bytes):
                audio_data.extend(message)
                # Print a dot for each audio chunk received for visual feedback
                sys.stdout.write(".")
                sys.stdout.flush()
                
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"\nConnection closed: {e}")
    except Exception as e:
        logger.error(f"\nError in receive_messages: {e}")
        
    print() # newline
    return audio_data

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client_headless.py <path_to_audio.wav>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    uri = "ws://127.0.0.1:8001/ws"
    
    logger.info(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected!")
            
            # Start receiving messages in the background
            receive_task = asyncio.create_task(receive_messages(websocket))
            
            # Send the audio file and silence
            await send_audio(websocket, file_path)
            
            # Wait for the receive task to finish (either connection closed or timeout)
            audio_data = await receive_task
            
            if audio_data:
                out_filename = "answer.wav"
                logger.info(f"Saving {len(audio_data)} bytes of audio to {out_filename}...")
                with wave.open(out_filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(24000)
                    wf.writeframes(audio_data)
                logger.info(f"Successfully saved to {out_filename}.")
            else:
                logger.warning("No audio data was received from the server.")
                
            # Close connection gracefully
            logger.info("Closing connection...")
            await websocket.close()
            
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running at {uri}?")
    except Exception as e:
        logger.error(f"Failed to connect or communicate: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
