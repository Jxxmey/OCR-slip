import re
from datetime import datetime
from typing import Optional
import os

# Import for CORS
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware

from pydantic import BaseModel
from google.cloud import vision
from dotenv import load_dotenv # Import load_dotenv
import pytz # For time zone awareness, you'll need to pip install pytz

# Load environment variables from .env file
# This will load variables from .env if running locally.
# On Render, it will use environment variables set directly in the dashboard.
load_dotenv()

# --- Debugging Google Cloud Vision Client Initialization ---
vision_client = None
try:
    # Attempt to initialize Google Cloud Vision client
    vision_client = vision.ImageAnnotatorClient()
    print("Google Cloud Vision client initialized successfully.")
except Exception as e:
    # Print a more detailed error for debugging on Render logs
    print(f"CRITICAL ERROR: Failed to initialize Google Cloud Vision client: {e}")
    print("Please ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
    print("It should be the Base64 encoded JSON key file content.")
    # We still set vision_client to None, and the perform_ocr function will raise a RuntimeError
    vision_client = None

app = FastAPI(
    title="Slip OCR and Parsing API",
    description="API for performing OCR on slip images and extracting key information like amount, date, and reference number.",
    version="1.0.0"
)

# --- CORS Middleware Configuration ---
# Configure CORS to allow requests from your frontend.
# If your frontend is deployed on Render, add its URL here.
# For local development, 'http://localhost' and specific ports are common.
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "https://ocr-slip.onrender.com", # Replace with your actual Render service URL if different
    # If you have a separate frontend hosted on Render or elsewhere, add its domain here:
    # "https://your-frontend-app.onrender.com",
    # "https://www.your-custom-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- OCR Function ---
async def perform_ocr(image_bytes: bytes) -> str:
    """Sends an image to Google Cloud Vision API to extract text."""
    if vision_client is None:
        # This error is raised if the client failed to initialize at startup
        raise RuntimeError("Google Cloud Vision client is not initialized. Please check GOOGLE_APPLICATION_CREDENTIALS environment variable on Render logs.")
    
    image = vision.Image(content=image_bytes)
    try:
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description
        return ""
    except Exception as e:
        # Catch specific Vision API errors and re-raise as HTTPException
        raise HTTPException(status_code=500, detail=f"Google Cloud Vision API error during text detection: {e}")

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
                # Handle both comma and dot as decimal/thousand separators
                num_str = match.group(1).replace(',', '').replace('.', '@').replace('@', '.')
                if num_str.count('.') > 1: # If multiple dots (e.g., 1.000.00), assume it's a thousand separator
                    num_str = num_str.replace('.', '')
                
                amount = float(num_str)
                if amount > 0.99: # Filter out very small numbers that might be noise
                    break
                else:
                    amount = None # Reset if it's too small
            except ValueError:
                amount = None # Keep amount as None if conversion fails

    if amount is None: # Fallback for amounts without keywords
        amount_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2}))', # e.g., 1,234.56
            r'(\d+\.\d{2})',                      # e.g., 123.45
            r'(\d{1,3}(?:,\d{3})*)',              # e.g., 1,234 (no decimals)
            r'(\d+)'                              # e.g., 123 (integers only)
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

    # Thai month mapping for date parsing
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

    # Date and Time Patterns
    datetime_patterns = [
        r'(\d{2}[-/]\d{2}[-/]\d{4}\s+\d{2}:\d{2}:\d{2})', # DD-MM-YYYY HH:MM:SS
        r'(\d{2}[-/]\d{2}[-/]\d{4}\s+\d{2}:\d{2})',     # DD-MM-YYYY HH:MM
        r'(\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})', # DD-MM-YY HH:MM:SS
        r'(\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2})',     # DD-MM-YY HH:MM
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+\d{2}:\d{2})', # D Mon YYYY HH:MM
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2}\s+\d{2}:\d{2})', # D Mon YY HH:MM
        r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})', # YYYY-MM-DD HH:MM:SS
        r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2})',     # YYYY-MM-DD HH:MM
        r'(\d{2}[-/]\d{2}[-/]\d{4})',                 # DD-MM-YYYY
        r'(\d{2}[-/]\d{2}[-/]\d{2})',                 # DD-MM-YY
        r'(\d{2}:\d{2}:\d{2})',                       # HH:MM:SS (time only)
        r'(\d{2}:\d{2})'                              # HH:MM (time only)
    ]

    # Date Formats to try parsing
    date_formats = [
        "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
        "%d-%m-%y %H:%M:%S", "%d/%m/%y %H:%M:%S",
        "%d/%m/%y %H:%M", "%d-%m-%y %H:%M",
        "%d %b %Y %H:%M", "%d %b %y %H:%M", # E.g., 29 Jul 2025 09:20
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%d-%m-%Y", "%d/%m/%Y",
        "%d-%m-%y", "%d/%m/%y",
        "%H:%M:%S", "%H:%M" # Time only formats
    ]

    return_date_time = None
    for pattern in datetime_patterns:
        match = re.search(pattern, processed_text_for_date, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            for fmt in date_formats:
                try:
                    # Handle Buddhist year (e.g., 2568) conversion to Gregorian (2025)
                    if '%Y' in fmt:
                        year_match = re.search(r'\d{4}', date_str)
                        if year_match and len(year_match.group(0)) == 4:
                            year_in_str = int(year_match.group(0))
                            if year_in_str > 2500: # Assuming year > 2500 is likely Buddhist year
                                date_str_gregorian = date_str.replace(str(year_in_str), str(year_in_str - 543))
                                date_time = datetime.strptime(date_str_gregorian, fmt)
                                return_date_time = date_time
                                break # Found a full date, so break from inner loop
                    
                    # Try parsing with current format
                    date_time = datetime.strptime(date_str, fmt)
                    return_date_time = date_time
                    break # Found a full date, so break from inner loop
                except ValueError:
                    continue # Try next format
            if return_date_time:
                # If only time was parsed (year=1900, month=1, day=1),
                # combine it with the current date in Bangkok.
                if return_date_time.year == 1900 and return_date_time.month == 1 and return_date_time.day == 1:
                    bangkok_tz = pytz.timezone('Asia/Bangkok')
                    current_time_bangkok = datetime.now(bangkok_tz)
                    return_date_time = current_time_bangkok.replace(
                        hour=return_date_time.hour,
                        minute=return_date_time.minute,
                        second=return_date_time.second,
                        microsecond=0,
                        tzinfo=None # Remove timezone info for Pydantic (if not needed)
                    )
                break # Break from outer loop once a date_time is found

    date_time = return_date_time

    # Reference Number Patterns
    ref_patterns = [
        r'(?:Ref\s*|Reference\s*|เลขที่อ้างอิง\s*|Ref No\.\s*|TRAN ID:\s*|TRN ID:\s*|Trx Ref:\s*|TRN\s*|Txn\s*|Transaction No\.\s*|หมายเลขอ้างอิง\s*|รหัสอ้างอิง\s*|รหัสรายการ\s*|หมายเลขรายการ\s*|เลขที่อ้างอิงรายการ\s*)(\S{8,40})', 
        r'(\d{10,30})', # Long sequence of digits (e.g., bank transaction IDs)
        r'(?:R\s*|TID\s*|Tran ID\s*|Ref\s*)\s*(\d{6,25})', # Shorter common patterns
        r'([A-Z0-9]{8,40})' # Alphanumeric sequences (general fallback)
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, lower_text, re.IGNORECASE)
        if match:
            reference_no = match.group(1).strip()
            
            # Add a check to prevent parsing dates/times or amounts as reference numbers
            is_date_or_time = False
            temp_dt_str = reference_no.replace('/', '-').replace(' ', ' ') 
            for fmt in date_formats:
                try:
                    if '%Y' in fmt:
                        year_in_ref_match = re.search(r'\d{4}', temp_dt_str)
                        if year_in_ref_match and len(year_in_ref_match.group(0)) == 4:
                            year_in_ref = int(year_in_ref_match.group(0))
                            if year_in_ref > 2500: # Apply Buddhist year conversion for ref check too
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
                    pass # Keep is_amount as False if parse_simple_amount fails

                if not is_amount:
                    # If it's not a date/time AND not an amount, then it's likely a reference number
                    break # Break from ref_patterns loop once a valid ref_no is found
                else:
                    reference_no = None # It was an amount, so not a ref_no
            else:
                reference_no = None # It was a date/time, so not a ref_no

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
            # If multiple dots (e.g., 1.000.00), assume the last part is decimal
            if num_str.count('.') > 1:
                parts = num_str.split('.')
                num_str = "".join(parts[:-1]) + "." + parts[-1]
            
            amount = float(num_str)
            if amount > 0:
                return amount
        except ValueError:
            return None
    
    # Fallback if no currency keywords are found but it looks like a number
    amount_match_fallback = re.search(r'^\s*(\d+(?:\.\d+)?)\s*$', text.strip())
    if amount_match_fallback:
        try:
            amount = float(amount_match_fallback.group(1))
            if amount > 0:
                return amount
        except ValueError:
            return None
            
    return None

# --- Pydantic Models for API Response ---
class ParsedSlipResponse(BaseModel):
    amount: Optional[float]
    date_time: Optional[datetime]
    reference_no: Optional[str]
    raw_text: Optional[str] = None # Added for debugging/info

# --- API Endpoints ---
@app.post("/parse-slip-image", response_model=ParsedSlipResponse, summary="Perform OCR on an image and parse slip information")
async def parse_slip_image(file: UploadFile = File(...)):
    """
    Receives a slip image file (PNG, JPG), performs OCR to extract text,
    then analyzes the text to extract amount, date/time, and reference number.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    image_bytes = await file.read()
    
    # Perform OCR
    try:
        raw_text = await perform_ocr(image_bytes)
    except RuntimeError as e:
        # This will catch the error if vision_client was not initialized
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException as e:
        # Re-raise Google Cloud Vision API errors (e.g., network issues)
        raise e 
    except Exception as e:
        # Catch any other unexpected OCR errors
        raise HTTPException(status_code=500, detail=f"Failed to perform OCR: {e}")

    # Parse the extracted text
    parsed_data = parse_slip_text(raw_text)

    return ParsedSlipResponse(
        amount=parsed_data["amount"],
        date_time=parsed_data["date_time"],
        reference_no=parsed_data["reference_no"],
        raw_text=raw_text # Include raw text for verification
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