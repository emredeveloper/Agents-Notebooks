import sys
import os
import fitz  # PyMuPDF
from ollama import chat
import re
from PIL import Image, ImageDraw, ImageFont

def get_pdf_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return input('Please enter the path to the PDF file: ')

def save_output(content, prompt_name, page_num, extension="txt"):
    filename = f"output_page_{page_num}_{prompt_name.lower().replace(' ', '_')}.{extension}"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Output saved to: {os.path.abspath(filename)}")
    return filename

def visualize_grounding(image_path, content, prompt_name, page_num):
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(.*?)\]\]<\|/det\|>'
        matches = re.findall(pattern, content)
        
        if not matches:
            return

        print(f"    [+] Generating visual output for Page {page_num} ({len(matches)} items)...")
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        colors = ["red", "blue", "green", "orange", "purple", "cyan"]
        
        for i, (label, coords_str) in enumerate(matches):
            try:
                coords = [int(x.strip()) for x in coords_str.split(',')]
                if len(coords) == 4:
                    x1 = coords[0] / 1000 * width
                    y1 = coords[1] / 1000 * height
                    x2 = coords[2] / 1000 * width
                    y2 = coords[3] / 1000 * height
                    
                    color = colors[i % len(colors)]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
                    text_w = right - left
                    text_h = bottom - top
                    
                    label_y = y1 - text_h - 4
                    if label_y < 0: label_y = y1
                    
                    draw.rectangle([x1, label_y, x1 + text_w + 4, label_y + text_h + 4], fill=color)
                    draw.text((x1 + 2, label_y), label, fill="white", font=font)
            except ValueError:
                continue

        output_filename = f"visualized_page_{page_num}_{prompt_name.lower().replace(' ', '_')}.png"
        img.save(output_filename)
        print(f"    [+] Visualized image saved to: {os.path.abspath(output_filename)}")
        
    except Exception as e:
        print(f"    [!] Error during visualization: {e}")

def extract_images_from_pdf(pdf_path):
    print(f"\n[i] Extracting images from PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    image_paths = []
    
    output_dir = "pdf_pages"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        image_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        
    print(f"[i] Extracted {len(image_paths)} pages to '{output_dir}' directory.")
    return image_paths

def main():
    pdf_path = get_pdf_path()
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    # Extract images first
    try:
        page_images = extract_images_from_pdf(pdf_path)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        print("Make sure you have pymupdf installed: pip install pymupdf")
        return

    prompts = {
        "1": {"name": "Layout Analysis", "prompt": "<|grounding|>Given the layout of the image.", "ext": "txt"},
        "2": {"name": "Free OCR", "prompt": "Free OCR.", "ext": "txt"},
        "3": {"name": "Parse Figure", "prompt": "Parse the figure.", "ext": "md"},
        "4": {"name": "Extract Text", "prompt": "Extract the text in the image.", "ext": "txt"},
        "5": {"name": "Markdown Conversion", "prompt": "<|grounding|>Convert the document to markdown.", "ext": "md"}
    }

    while True:
        print("\n--- DeepSeek-OCR PDF Menu ---")
        print(f"PDF: {pdf_path} ({len(page_images)} pages)")
        for key, val in prompts.items():
            print(f"{key}. {val['name']}")
        print("q. Quit")
        
        choice = input("\nSelect a prompt (1-5) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            break
            
        if choice in prompts:
            selected = prompts[choice]
            print(f"\n--- Running: {selected['name']} on ALL pages ---")
            
            for i, img_path in enumerate(page_images):
                page_num = i + 1
                print(f"\nProcessing Page {page_num}...")
                
                try:
                    response = chat(
                        model='deepseek-ocr:latest',
                        messages=[
                            {
                                'role': 'user',
                                'content': selected['prompt'],
                                'images': [img_path],
                            }
                        ],
                    )
                    content = response.message.content
                    print(f"--- Output Page {page_num} (Preview) ---")
                    print(content[:200] + "..." if len(content) > 200 else content)
                    
                    # Save output
                    save_output(content, selected['name'], page_num, selected['ext'])
                    
                    # Visualization
                    if "<|grounding|>" in selected['prompt']:
                        visualize_grounding(img_path, content, selected['name'], page_num)
                        
                except Exception as e:
                    print(f"Error on page {page_num}: {e}")
                    
            print("\n" + "="*50 + "\n")
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()
