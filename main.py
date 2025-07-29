import re
from datetime import datetime
from typing import Optional
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware

from pydantic import BaseModel
from google.cloud import vision
from dotenv import load_dotenv
import pytz
import json # Import the json module

# Load environment variables from .env file
load_dotenv()

# --- Google Cloud Vision Client Initialization ---
vision_client = None
try:
    # Get the base64 encoded credentials string from the environment variable
    # Render will provide this directly, .env will provide it locally
    credentials_base64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64") 
    # NOTE: I've changed the variable name here to avoid confusion.
    # We will use GOOGLE_APPLICATION_CREDENTIALS_BASE64 in Render.

    if credentials_base64:
        # Decode the Base64 string to get the JSON content
        import base64
        credentials_json_bytes = base64.b64decode(credentials_base64)
        credentials_json_string = credentials_json_bytes.decode('utf-8')
        
        # Load the JSON content into a dictionary
        credentials_dict = json.loads(credentials_json_string)
        
        # Initialize the client from the dictionary (which represents the service account info)
        # This is the key change!
        vision_client = vision.ImageAnnotatorClient.from_service_account_info(credentials_dict)
        print("Google Cloud Vision client initialized successfully from Base64 credentials.")
    else:
        # Fallback for local development if you have GOOGLE_APPLICATION_CREDENTIALS set as a path
        # Or if no credentials_base64 is found, it will try default ADC
        vision_client = vision.ImageAnnotatorClient() 
        print("Google Cloud Vision client initialized using default credentials (GOOGLE_APPLICATION_CREDENTIALS filepath or ADC).")

except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Google Cloud Vision client: {e}")
    print("Please ensure GOOGLE_APPLICATION_CREDENTIALS_BASE64 (or GOOGLE_APPLICATION_CREDENTIALS) environment variable is set correctly.")
    print("For Render, use GOOGLE_APPLICATION_CREDENTIALS_BASE64 with the Base64 encoded JSON content.")
    vision_client = None

app = FastAPI(
    title="Slip OCR and Parsing API",
    description="API for performing OCR on slip images and extracting key information like amount, date, and reference number.",
    version="1.0.0"
)

# --- CORS Middleware Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "https://ocr-slip.onrender.com", # Replace with your actual Render service URL if different
    "https://jxxmey.github.io",
    "https://127.0.0.1",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OCR Function ---
import asyncio

async def perform_ocr(image_bytes: bytes) -> str:
    if vision_client is None:
        raise RuntimeError("Google Cloud Vision client is not initialized.")
    
    loop = asyncio.get_event_loop()
    image = vision.Image(content=image_bytes)
    try:
        response = await loop.run_in_executor(None, vision_client.text_detection, image)
        texts = response.text_annotations
        if texts:
            return texts[0].description
        return ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Cloud Vision API error: {e}")


# --- Parsing Functions ---
def parse_slip_text(text: str) -> dict:
    """Analyzes text from a slip to extract key information: amount, date/time, reference number."""
    amount = None
    date_time = None
    reference_no = None

    lower_text = text.lower()

    # Enhanced Amount Parsing
    amount_keywords_patterns = [
        r'(?:total|amount|รวม|ยอด|ชำระ|เป็นเงิน)\D*?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)',
        r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\D*?(?:บาท|baht|thb|total|amount|รวม|ยอด|ชำระ|เป็นเงิน)'
    ]

    for pattern in amount_keywords_patterns:
        match = re.search(pattern, lower_text, re.IGNORECASE)
        if match:
            try:
                num_str = match.group(1).replace(',', '').replace('.', '@').replace('@', '.')
                if num_str.count('.') > 1:
                    num_str = num_str.replace('.', '')
                
                amount = float(num_str)
                if amount > 0.99:
                    break
                else:
                    amount = None
            except ValueError:
                amount = None

    if amount is None:
        amount_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2}))',
            r'(\d+\.\d{2})',
            r'(\d{1,3}(?:,\d{3})*)',
            r'(\d+)'
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, lower_text)
            if match:
                try:
                    num_str = match.group(1).replace(',', '')
                    amount = float(num_str)
                    if amount > 0.99:
                        break
                    else:
                        amount = None
                except ValueError:
                    amount = None

    thai_month_map = {
        'ม.ค.': 'Jan', 'ก.พ.': 'Feb', 'มี.ค.': 'Mar', 'เม.ย.': 'Apr',
        'พ.ค.': 'May', 'มิ.ย.': 'Jun', 'ก.ค.': 'Jul', 'ส.ค.': 'Aug',
        'ก.ย.': 'Sep', 'ต.ค.': 'Oct', 'พ.ย.': 'Nov', 'ธ.ค.': 'Dec',
        'มกราคม': 'January', 'กุมภาพันธ์': 'February', 'มีนาคม': 'March', 'เมษายน': 'April',
        'พฤษภาคม': 'May', 'มิถุนายน': 'June', 'กรกฎาคม': 'July', 'สิงหาคม': 'August',
        'กันยายน': 'September', 'ตุลาคม': 'October', 'พฤศจิกายน': 'November', 'ธันวาคม': 'December'
    }

    processed_text_for_date = text
    for thai_m, eng_m in thai_month_map.items():
        processed_text_for_date = re.sub(re.escape(thai_m), eng_m, processed_text_for_date, flags=re.IGNORECASE)

    datetime_patterns = [
        r'(\d{2}[-/]\d{2}[-/]\d{4}\s+\d{2}:\d{2}:\d{2})',
        r'(\d{2}[-/]\d{2}[-/]\d{4}\s+\d{2}:\d{2})',
        r'(\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})',
        r'(\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+\d{2}:\d{2})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2}\s+\d{2}:\d{2})',
        r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})',
        r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2})',
        r'(\d{2}[-/]\d{2}[-/]\d{4})',
        r'(\d{2}[-/]\d{2}[-/]\d{2})',
        r'(\d{2}:\d{2}:\d{2})',
        r'(\d{2}:\d{2})'
    ]

    date_formats = [
        "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
        "%d-%m-%y %H:%M:%S", "%d/%m/%y %H:%M:%S",
        "%d/%m/%y %H:%M", "%d-%m-%y %H:%M",
        "%d %b %Y %H:%M", "%d %b %y %H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%d-%m-%Y", "%d/%m/%Y",
        "%d-%m-%y", "%d/%m/%y",
        "%H:%M:%S", "%H:%M"
    ]

    return_date_time = None
    for pattern in datetime_patterns:
        match = re.search(pattern, processed_text_for_date, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            for fmt in date_formats:
                try:
                    if '%Y' in fmt:
                        year_match = re.search(r'\d{4}', date_str)
                        if year_match and len(year_match.group(0)) == 4:
                            year_in_str = int(year_match.group(0))
                            if year_in_str > 2500:
                                date_str_gregorian = date_str.replace(str(year_in_str), str(year_in_str - 543))
                                date_time = datetime.strptime(date_str_gregorian, fmt)
                                return_date_time = date_time
                                break
                    
                    date_time = datetime.strptime(date_str, fmt)
                    return_date_time = date_time
                    break
                except ValueError:
                    continue
            if return_date_time:
                if return_date_time.year == 1900 and return_date_time.month == 1 and return_date_time.day == 1:
                    bangkok_tz = pytz.timezone('Asia/Bangkok')
                    current_time_bangkok = datetime.now(bangkok_tz)
                    return_date_time = current_time_bangkok.replace(
                        hour=return_date_time.hour,
                        minute=return_date_time.minute,
                        second=return_date_time.second,
                        microsecond=0,
                        tzinfo=None
                    )
                break
    
    date_time = return_date_time

    ref_patterns = [
        r'(?:Ref\s*|Reference\s*|เลขที่อ้างอิง\s*|Ref No\.\s*|TRAN ID:\s*|TRN ID:\s*|Trx Ref:\s*|TRN\s*|Txn\s*|Transaction No\.\s*|หมายเลขอ้างอิง\s*|รหัสอ้างอิง\s*|รหัสรายการ\s*|หมายเลขรายการ\s*|เลขที่อ้างอิงรายการ\s*)(\S{8,40})', 
        r'(\d{10,30})', 
        r'(?:R\s*|TID\s*|Tran ID\s*|Ref\s*)\s*(\d{6,25})',
        r'([A-Z0-9]{8,40})' 
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, lower_text, re.IGNORECASE)
        if match:
            reference_no = match.group(1).strip()
            is_date_or_time = False
            temp_dt_str = reference_no.replace('/', '-').replace(' ', ' ') 
            for fmt in date_formats:
                try:
                    if '%Y' in fmt:
                        year_in_ref_match = re.search(r'\d{4}', temp_dt_str)
                        if year_in_ref_match and len(year_in_ref_match.group(0)) == 4:
                            year_in_ref = int(year_in_ref_match.group(0))
                            if year_in_ref > 2500:
                                temp_dt_str = temp_dt_str.replace(str(year_in_ref), str(year_in_ref - 543))

                    datetime.strptime(temp_dt_str, fmt)
                    is_date_or_time = True
                    break
                except (ValueError, AttributeError):
                    continue
            
            if not is_date_or_time:
                is_amount = False
                try:
                    if parse_simple_amount(reference_no) is not None:
                        is_amount = True
                except:
                    pass

                if not is_amount:
                    break
                else:
                    reference_no = None
            else:
                reference_no = None

    return {
        "amount": amount,
        "date_time": date_time,
        "reference_no": reference_no
    }

def parse_simple_amount(text: str) -> float | None:
    """Attempts to extract a number representing an amount from user-provided text."""
    amount_match = re.search(r'^\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*(?:บาท|baht|thb|$)', text.strip(), re.IGNORECASE)
    if amount_match:
        try:
            num_str = amount_match.group(1).replace(',', '')
            if num_str.count('.') > 1:
                parts = num_str.split('.')
                num_str = "".join(parts[:-1]) + "." + parts[-1]
            
            amount = float(num_str)
            if amount > 0:
                return amount
        except ValueError:
            return None
    
    amount_match_fallback = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', text.strip())
    if amount_match_fallback:
        try:
            amount = float(amount_match_fallback.group(1))
            if amount > 0:
                return amount
        except ValueError:
            return None
            
    return None

class ParsedSlipResponse(BaseModel):
    amount: Optional[float]
    date_time: Optional[datetime]
    reference_no: Optional[str]
    raw_text: Optional[str] = None

@app.post("/parse-slip-image", response_model=ParsedSlipResponse, summary="Perform OCR on an image and parse slip information")
async def parse_slip_image(file: UploadFile = File(...)):
    """
    Receives a slip image file (PNG, JPG), performs OCR to extract text,
    then analyzes the text to extract amount, date/time, and reference number.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    image_bytes = await file.read()
    
    try:
        raw_text = await perform_ocr(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform OCR: {e}")

    parsed_data = parse_slip_text(raw_text)

    return ParsedSlipResponse(
        amount=parsed_data["amount"],
        date_time=parsed_data["date_time"],
        reference_no=parsed_data["reference_no"],
        raw_text=raw_text
    )

class ParseTextRequest(BaseModel):
    text: str

@app.post("/parse-slip-text", response_model=ParsedSlipResponse, summary="Parse slip information from raw text")
async def parse_slip_text_direct(request: ParseTextRequest):
    """
    Receives raw text presumed to be from a slip and analyzes it to extract
    amount, date/time, and reference number.
    """
    parsed_data = parse_slip_text(request.text)
    return ParsedSlipResponse(
        amount=parsed_data["amount"],
        date_time=parsed_data["date_time"],
        reference_no=parsed_data["reference_no"],
        raw_text=request.text
    )