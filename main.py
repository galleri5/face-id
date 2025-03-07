import json
import logging
import traceback
import requests
import time
import uuid
import base64
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face-text-generator")

app = FastAPI(title="Face Text Generator API")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# ComfyUI API URLs
SERVER_ADDRESS = "0.0.0.0:6000"
COMFY_HTTP_URL = f"http://{SERVER_ADDRESS}"
COMFY_PROMPT_URL = f"{COMFY_HTTP_URL}/prompt"
COMFY_WS_URL = f"ws://{SERVER_ADDRESS}/ws"
COMFY_VIEW_URL = f"{COMFY_HTTP_URL}/view"


# Input model
class GenerationRequest(BaseModel):
    base64_image: str
    prompt: Optional[str] = "MS Paint art aesthetic hyper minimalist line art style Ultra minimalist cartoon"


# Check if ComfyUI is available
def check_comfyui_health():
    try:
        # Try to connect to the root endpoint
        response = requests.get(COMFY_HTTP_URL, timeout=5)
        logger.info(f"ComfyUI health check status: {response.status_code}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.warning(f"ComfyUI health check failed: {str(e)}")
        return False


# Startup event to wait for ComfyUI to be ready
@app.on_event("startup")
async def startup_event():
    logger.info("Waiting for ComfyUI to be ready...")
    max_attempts = 30
    for attempt in range(max_attempts):
        if check_comfyui_health():
            logger.info("ComfyUI is ready")
            return
        logger.info(f"ComfyUI not ready yet, attempt {attempt+1}/{max_attempts}")
        time.sleep(2)
    logger.warning("Could not confirm ComfyUI is ready, but continuing anyway")


def queue_prompt(prompt: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """Queue a prompt in ComfyUI and return the response"""
    try:
        payload = {"prompt": prompt, "client_id": client_id}
        logger.debug(f"Sending prompt payload: {json.dumps(payload)[:500]}...")
        
        response = requests.post(
            COMFY_PROMPT_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            error_msg = f"Failed to queue prompt: HTTP {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        return response.json()
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error in queue_prompt: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


def get_image_from_comfyui(filename, subfolder=""):
    """Download an image from ComfyUI and return it as base64"""
    try:
        # Build the URL to fetch the image
        url = f"{COMFY_VIEW_URL}?filename={filename}&subfolder={subfolder}&type=output"
        logger.info(f"Fetching image from: {url}")
        
        # Download the image
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch image: HTTP {response.status_code}")
            return None
        
        # Convert to base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        logger.info(f"Image converted to base64 (length: {len(image_base64)})")
        
        return image_base64
    except Exception as e:
        logger.error(f"Error fetching image {filename}: {str(e)}")
        return None


def process_comfyui_generation(prompt, client_id):
    """Process the generation through ComfyUI"""
    try:
        # Check if websocket-client package is installed
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client package not installed. Install with: pip install websocket-client")
            raise HTTPException(
                status_code=500, 
                detail="Server configuration error: websocket-client package not installed"
            )
        
        # Queue the prompt
        logger.info("Queueing prompt...")
        queue_response = queue_prompt(prompt, client_id)
        prompt_id = queue_response.get("prompt_id")
        
        if not prompt_id:
            error_msg = f"No prompt_id received from ComfyUI: {queue_response}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Prompt queued with ID: {prompt_id}")
        
        # Connect to WebSocket to monitor progress
        ws_url = f"{COMFY_WS_URL}?clientId={client_id}"
        logger.info(f"Connecting to WebSocket: {ws_url}")
        
        ws = websocket.create_connection(ws_url)
        logger.info("WebSocket connection established")
        
        # Variables to track progress
        image_filenames = []
        execution_complete = False
        max_wait_time = 300  # Maximum wait time in seconds
        start_time = time.time()
        
        try:
            # Monitor execution progress
            while time.time() - start_time < max_wait_time and not execution_complete:
                try:
                    # Set timeout for receiving messages
                    ws.settimeout(5.0)
                    message = ws.recv()
                    
                    # Process text messages
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type", "")
                            
                            # Log progress
                            if msg_type == "progress":
                                progress = data.get("data", {})
                                if progress.get("prompt_id") == prompt_id:
                                    logger.info(f"Progress: {progress.get('value', 0)}/{progress.get('max', 100)}")
                            
                            # Log execution
                            elif msg_type == "executing":
                                exec_data = data.get("data", {})
                                if exec_data.get("prompt_id") == prompt_id:
                                    node = exec_data.get("node")
                                    if node is None:
                                        logger.info("Execution completed")
                                        execution_complete = True
                                    else:
                                        logger.info(f"Executing node: {node}")
                            
                            # Look for executed messages with image information
                            elif msg_type == "executed":
                                exec_data = data.get("data", {})
                                logger.debug(f"Executed message: {json.dumps(data)[:1000]}...")
                                
                                if exec_data.get("prompt_id") == prompt_id:
                                    # Look for image output in various formats
                                    output = exec_data.get("output", {})
                                    
                                    # Direct images array at the top level
                                    if "images" in output:
                                        new_images = output.get("images", [])
                                        logger.info(f"Found {len(new_images)} image references")
                                        image_filenames.extend(new_images)
                                    
                                    # Images in a node (like node 15)
                                    elif "15" in output and "images" in output["15"]:
                                        new_images = output["15"].get("images", [])
                                        logger.info(f"Found {len(new_images)} image references in node 15")
                                        image_filenames.extend(new_images)
                            
                            # Check for execution success
                            elif msg_type == "execution_success":
                                if data.get("data", {}).get("prompt_id") == prompt_id:
                                    logger.info("Execution successfully completed")
                                    execution_complete = True
                        
                        except json.JSONDecodeError:
                            pass
                    
                except websocket.WebSocketTimeoutException:
                    # Just a timeout, continue
                    continue
            
            # Check if we have image filenames
            if not image_filenames:
                logger.warning("No image filenames collected from WebSocket")
                
                # Try to get images from history
                try:
                    history_url = f"{COMFY_HTTP_URL}/history"
                    logger.info(f"Fetching history from {history_url}")
                    
                    history_response = requests.get(history_url, timeout=10)
                    if history_response.status_code == 200:
                        history_data = history_response.json()
                        
                        if prompt_id in history_data:
                            prompt_data = history_data[prompt_id]
                            outputs = prompt_data.get("outputs", {})
                            
                            # Look for images in outputs
                            for node_id, node_output in outputs.items():
                                if "images" in node_output:
                                    new_images = node_output["images"]
                                    logger.info(f"Found {len(new_images)} image references in history")
                                    image_filenames.extend(new_images)
                except Exception as e:
                    logger.error(f"Error fetching history: {str(e)}")
            
            # If we still don't have image filenames, try to list output directory
            if not image_filenames:
                try:
                    files_url = f"{COMFY_VIEW_URL}"
                    logger.info(f"Listing files from {files_url}")
                    
                    files_response = requests.get(files_url, timeout=10)
                    if files_response.status_code == 200:
                        files_data = files_response.json()
                        
                        # Look for TextFilter files
                        output_files = [f for f in files_data.get("files", []) 
                                      if f.startswith("TextFilter_")]
                        
                        if output_files:
                            logger.info(f"Found {len(output_files)} TextFilter files")
                            for filename in output_files:
                                image_filenames.append({
                                    "filename": filename,
                                    "subfolder": "",
                                    "type": "output"
                                })
                except Exception as e:
                    logger.error(f"Error listing files: {str(e)}")
        
        finally:
            # Close the WebSocket connection
            ws.close()
            logger.info("WebSocket connection closed")
        
        # Now download and convert each image
        if not image_filenames:
            logger.error("No image filenames found")
            raise ValueError("No images were generated")
        
        result_images = []
        for idx, img_info in enumerate(image_filenames):
            # Handle both dictionary and string formats
            if isinstance(img_info, dict):
                filename = img_info.get("filename")
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")
            else:
                filename = img_info
                subfolder = ""
                img_type = "output"
            
            # Skip if no filename
            if not filename:
                continue
            
            # Download and convert the image
            image_base64 = get_image_from_comfyui(filename, subfolder)
            
            if image_base64:
                result_images.append({
                    "filename": filename,
                    "image": image_base64,
                    "type": img_type
                })
        
        if not result_images:
            logger.error("No images could be downloaded")
            raise ValueError("Failed to download any images")
        
        return {
            "status": "success",
            "prompt_id": prompt_id,
            "images": result_images
        }
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in process_comfyui_generation: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


# Function to generate face text
def face_text_gen(f64, prompt):
    logger.info("Starting face_text_gen function")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Base64 image length: {len(f64) if f64 else 0}")
    
    # Create a unique client ID for this request
    client_id = str(uuid.uuid4())
    logger.info(f"Generated client ID: {client_id}")
    
    # Construct the payload for ComfyUI
    workflow = {
        "3": {
            "inputs": {
                "seed": 961893869568887,
                "steps": 5,
                "cfg": 2,
                "sampler_name": "dpmpp_sde_gpu",
                "scheduler": "karras",
                "denoise": 1,
                "model": ["60", 0],
                "positive": ["60", 1],
                "negative": ["60", 2],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "4": {
            "inputs": {"ckpt_name": "juggLight.safetensors"},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "5": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"}
        },
        "8": {
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "11": {
            "inputs": {"instantid_file": "ip-instant.bin"},
            "class_type": "InstantIDModelLoader",
            "_meta": {"title": "Load InstantID Model"}
        },
        "15": {
            "inputs": {"filename_prefix": "TextFilter_", "images": ["8", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        },
        "16": {
            "inputs": {"control_net_name": "instantid/diffusion_pytorch_model.safetensors"},
            "class_type": "ControlNetLoader",
            "_meta": {"title": "Load ControlNet Model"}
        },
        "38": {
            "inputs": {"provider": "CPU"},
            "class_type": "InstantIDFaceAnalysis",
            "_meta": {"title": "InstantID Face Analysis"}
        },
        "39": {
            "inputs": {"text": prompt, "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"}
        },
        "40": {
            "inputs": {
                "text": "text, alphabet, naked, nipples, photograph, fingers, deformed, glitch, noisy, realistic, stock photo",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"}
        },
        "60": {
            "inputs": {
                "weight": 0.8,
                "start_at": 0,
                "end_at": 1,
                "instantid": ["11", 0],
                "insightface": ["38", 0],
                "control_net": ["16", 0],
                "image": ["61", 0],
                "model": ["4", 0],
                "positive": ["39", 0],
                "negative": ["40", 0]
            },
            "class_type": "ApplyInstantID",
            "_meta": {"title": "Apply InstantID"}
        },
        "61": {
            "inputs": {"image": f64},
            "class_type": "ETN_LoadImageBase64",
            "_meta": {"title": "Load Image (Base64)"}
        }
    }
    
    try:
        # Verify ComfyUI is responsive before sending request
        if not check_comfyui_health():
            logger.error("ComfyUI is not responding")
            raise HTTPException(status_code=503, detail="ComfyUI service unavailable")
        
        # Process the generation with ComfyUI
        return process_comfyui_generation(workflow, client_id)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/generate")
async def generate(request: GenerationRequest, req: Request):
    """
    Generate stylized face image based on input image and prompt
    
    - **base64_image**: Base64 encoded image string
    - **prompt**: Text prompt to guide the generation (optional)
    """
    logger.info("Received generation request")
    
    # Log client information
    client_host = req.client.host if req.client else "unknown"
    logger.info(f"Request from: {client_host}")
    logger.info(f"Prompt length: {len(request.prompt) if request.prompt else 0}")
    logger.info(f"Base64 image length: {len(request.base64_image) if request.base64_image else 0}")
    
    try:
        result = face_text_gen(request.base64_image, request.prompt)
        logger.info("Request completed successfully")
        return result
    except HTTPException as e:
        logger.error(f"HTTP Exception occurred: {e.detail}")
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Health check requested")
    comfy_status = "up" if check_comfyui_health() else "down"
    return {
        "status": "online", 
        "message": "Face Text Generator API is running",
        "comfyui_status": comfy_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)