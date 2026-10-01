import uvicorn
import logging
from src.config import FRONTEND_DIR

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("================================================================")
    print("🦅 Kestrel Home Appliances - Returns Risk Intelligence Engine")
    print("================================================================")
    print("Starting server on http://127.0.0.1:8000")
    print("Interactive Operations Dashboard: http://127.0.0.1:8000")
    print("Swagger API Documentation:        http://127.0.0.1:8000/docs")
    print("================================================================")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
