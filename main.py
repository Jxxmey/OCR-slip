import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from google.cloud import vision
import os
from dotenv import load_dotenv # <-- NEW: Import load_dotenv

# Load environment variables from .env file
load_dotenv() # <-- NEW: Call load_dotenv to load variables

# Initialize Google Cloud Vision client
# Make sure GOOGLE_APPLICATION_CREDENTIALS environment variable is set
try:
    # The GOOGLE_APPLICATION_CREDENTIALS will now be loaded from your .env file
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Error initializing Google Cloud Vision client: {e}")
    print("Please ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly in your .env file or system environment.")
    vision_client = None

app = FastAPI(
    title="Slip OCR and Parsing API",
    description="API for performing OCR on slip images and extracting key information like amount, date/time, and reference number.",
    version="1.0.0"
)

# --- OCR Function ---
async def perform_ocr(image_bytes: bytes) -> str:
    """ส่งรูปภาพไปยัง Google Cloud Vision API เพื่อดึงข้อความ"""
    if vision_client is None:
        raise RuntimeError("Google Cloud Vision client is not initialized. Check GOOGLE_APPLICATION_CREDENTIALS.")
    
    image = vision.Image(content=image_bytes)
    try:
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description
        return ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Cloud Vision API error: {e}")

# --- Parsing Functions (from your provided code) ---
def parse_slip_text(text: str) -> dict:
    """วิเคราะห์ข้อความจากสลิปเพื่อดึงข้อมูลสำคัญ: จำนวนเงิน, วันที่-เวลา, เลขที่อ้างอิง"""
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
                # Handle cases where comma is used as thousands separator and dot as decimal
                # or vice-versa, then normalize to dot for float conversion
                num_str = match.group(1).replace(',', '').replace('.', '@').replace('@', '.')
                if num_str.count('.') > 1: # If multiple dots (e.g., 1.000.00)
                    num_str = num_str.replace('.', '') # Remove all dots (assuming they are thousands separators)
                
                amount = float(num_str)
                if amount > 0.99: # Ensure it's a plausible amount, not just a stray number like '0.50' (which might be part of a date or time)
                    break
                else:
                    amount = None
            except ValueError:
                amount = None

    if amount is None: # Fallback for amounts without keywords
        amount_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2}))', # e.g., 1,234.56
            r'(\d+\.\d{2})', # e.g., 123.45
            r'(\d{1,3}(?:,\d{3})*)', # e.g., 1,234
            r'(\d+)' # e.g., 123
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

    # --- Date/Time Parsing ---
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
        r'(\d{2}[-/]\d{2}[-/]\d{4}\s+\d{2}:\d{2}:\d{2})', # DD-MM-YYYY HH:MM:SS
        r'(\d{2}[-/]\d{2}[-/]\d{4}\s+\d{2}:\d{2})',       # DD-MM-YYYY HH:MM
        r'(\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})', # DD-MM-YY HH:MM:SS
        r'(\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2})',       # DD-MM-YY HH:MM
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+\d{2}:\d{2})', # D Mon YYYY HH:MM
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2}\s+\d{2}:\d{2})', # D Mon YY HH:MM
        r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})', # YYYY-MM-DD HH:MM:SS
        r'(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2})',       # YYYY-MM-DD HH:MM
        r'(\d{2}[-/]\d{2}[-/]\d{4})',                     # DD-MM-YYYY
        r'(\d{2}[-/]\d{2}[-/]\d{2})',                     # DD-MM-YY
        r'(\d{2}:\d{2}:\d{2})',                           # HH:MM:SS (assume current date if only time)
        r'(\d{2}:\d{2})'                                  # HH:MM (assume current date if only time)
    ]

    date_formats = [
        "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
        "%d-%m-%y %H:%M:%S", "%d/%m/%y %H:%M:%S",
        "%d/%m/%y %H:%M", "%d-%m-%y %H:%M",
        "%d %b %Y %H:%M", "%d %b %y %H:%M", # for English month abbreviations
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
                    # Handle Buddhist year (BE) to Gregorian year (AD) conversion if year > 2500
                    if '%Y' in fmt:
                        year_match = re.search(r'\d{4}', date_str)
                        if year_match and len(year_match.group(0)) == 4:
                            year_in_str = int(year_match.group(0))
                            if year_in_str > 2500: # Assuming 25xx is Buddhist year
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
                # If only time was extracted, set date to current date
                if return_date_time.year == 1900 and return_date_time.month == 1 and return_date_time.day == 1:
                    now = datetime.now() # Use current time in Bangkok
                    return_date_time = now.replace(
                        hour=return_date_time.hour,
                        minute=return_date_time.minute,
                        second=return_date_time.second,
                        microsecond=0
                    )
                break
    
    date_time = return_date_time

    # --- Reference Number Parsing ---
    ref_patterns = [
        r'(?:Ref\s*|Reference\s*|เลขที่อ้างอิง\s*|Ref No\.\s*|TRAN ID:\s*|TRN ID:\s*|Trx Ref:\s*|TRN\s*|Txn\s*|Transaction No\.\s*|หมายเลขอ้างอิง\s*|รหัสอ้างอิง\s*|รหัสรายการ\s*|หมายเลขรายการ\s*|เลขที่อ้างอิงรายการ\s*)(\S{8,40})', 
        r'(\d{10,30})', # Long sequence of digits (e.g., transaction ID)
        r'(?:R\s*|TID\s*|Tran ID\s*|Ref\s*)\s*(\d{6,25})', # Shorter ref numbers with keywords
        r'([A-Z0-9]{8,40})' # Alphanumeric patterns
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, lower_text, re.IGNORECASE)
        if match:
            reference_no = match.group(1).strip()
            # Validate if the found "reference_no" is actually a date/time or an amount
            is_date_or_time = False
            temp_dt_str = reference_no.replace('/', '-').replace(' ', ' ') # Normalize for date parsing
            for fmt in date_formats:
                try:
                    if '%Y' in fmt:
                        year_in_ref_match = re.search(r'\d{4}', temp_dt_str)
                        if year_in_ref_match and len(year_in_ref_match.group(0)) == 4:
                            year_in_ref = int(year_in_ref_match.group(0))
                            if year_in_ref > 2500: # Convert Buddhist year if applicable
                                temp_dt_str = temp_dt_str.replace(str(year_in_ref), str(year_in_ref - 543))

                    datetime.strptime(temp_dt_str, fmt)
                    is_date_or_time = True
                    break
                except (ValueError, AttributeError):
                    continue
            
            if not is_date_or_time: # If it's not a date/time, check if it's an amount
                is_amount = False
                try:
                    if parse_simple_amount(reference_no) is not None:
                        is_amount = True
                except:
                    pass

                if not is_amount: # If it's neither a date/time nor an amount, then it's likely a reference number
                    break
                else:
                    reference_no = None # It was an amount, so discard
            else:
                reference_no = None # It was a date/time, so discard

    return {
        "amount": amount,
        "date_time": date_time,
        "reference_no": reference_no
    }

def parse_simple_amount(text: str) -> float | None:
    """พยายามดึงตัวเลขที่เป็นจำนวนเงินจากข้อความที่ผู้ใช้พิมพ์เข้ามา"""
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

# --- Pydantic Models for API Response ---
class ParsedSlipResponse(BaseModel):
    amount: Optional[float]
    date_time: Optional[datetime]
    reference_no: Optional[str]
    raw_text: Optional[str] = None # Added for debugging/info

# --- API Endpoints ---
---
## Endpoint: `/parse-slip-image`

นี่คือ Endpoint ที่จะรับรูปภาพสลิปเป็นไฟล์และทำการ OCR พร้อมทั้งวิเคราะห์ข้อมูล

```python
@app.post("/parse-slip-image", response_model=ParsedSlipResponse, summary="Perform OCR on an image and parse slip information")
async def parse_slip_image(file: UploadFile = File(...)):
    """
    รับไฟล์รูปภาพสลิป (PNG, JPG) ทำการ OCR เพื่อดึงข้อความ
    จากนั้นวิเคราะห์ข้อความเพื่อดึงข้อมูลจำนวนเงิน, วันที่-เวลา, และเลขที่อ้างอิง
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    image_bytes = await file.read()
    
    # Perform OCR
    try:
        raw_text = await perform_ocr(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException as e:
        raise e # Re-raise Google Cloud Vision API errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform OCR: {e}")

    # Parse the extracted text
    parsed_data = parse_slip_text(raw_text)

    return ParsedSlipResponse(
        amount=parsed_data["amount"],
        date_time=parsed_data["date_time"],
        reference_no=parsed_data["reference_no"],
        raw_text=raw_text # Include raw text for verification
    )

---
## Endpoint: `/parse-slip-text`

Endpoint นี้จะรับข้อความที่ผู้ใช้ป้อนเข้ามาโดยตรง (ในกรณีที่มีข้อความอยู่แล้ว ไม่ต้องทำ OCR) และทำการวิเคราะห์ข้อมูล

```python
class ParseTextRequest(BaseModel):
    text: str

@app.post("/parse-slip-text", response_model=ParsedSlipResponse, summary="Parse slip information from raw text")
async def parse_slip_text_direct(request: ParseTextRequest):
    """
    รับข้อความดิบที่คาดว่าเป็นข้อความจากสลิป และวิเคราะห์เพื่อดึงข้อมูล
    จำนวนเงิน, วันที่-เวลา, และเลขที่อ้างอิง
    """
    parsed_data = parse_slip_text(request.text)
    return ParsedSlipResponse(
        amount=parsed_data["amount"],
        date_time=parsed_data["date_time"],
        reference_no=parsed_data["reference_no"],
        raw_text=request.text
    )