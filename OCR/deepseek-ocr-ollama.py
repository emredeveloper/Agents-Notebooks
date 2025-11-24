from ollama import chat
import sys
import os
import re

# Try to import PIL for image manipulation
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def get_image_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return input('Please enter the path to the image: ')

def save_output(content, prompt_name, extension="txt"):
    filename = f"output_{prompt_name.lower().replace(' ', '_')}.{extension}"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[+] Output saved to: {os.path.abspath(filename)}")
    return filename

def visualize_grounding(image_path, content, prompt_name):
    if not PIL_AVAILABLE:
        print("\n[!] Pillow library not found. Cannot generate visual output.")
        print("    Please run: pip install Pillow")
        return

    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Parse pattern: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
        # Regex explanation:
        # <\|ref\|>(.*?)<\|/ref\|>  -> Captures the label
        # <\|det\|>\[\[(.*?)\]\]<\|/det\|> -> Captures the coordinates string inside [[ ]]
        pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(.*?)\]\]<\|/det\|>'
        matches = re.findall(pattern, content)
        
        if not matches:
            print("\n[i] No coordinates found in the output to visualize.")
            return

        print(f"\n[+] Generating visual output for {len(matches)} items...")
        
        # Load a font (optional, fallback to default)
        try:
            # Try to find a standard font on Windows
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        colors = ["red", "blue", "green", "orange", "purple", "cyan"]
        
        for i, (label, coords_str) in enumerate(matches):
            try:
                # coords_str is like "35, 81, 557, 156"
                coords = [int(x.strip()) for x in coords_str.split(',')]
                
                if len(coords) == 4:
                    # DeepSeek-OCR coordinates are normalized 0-1000
                    x1 = coords[0] / 1000 * width
                    y1 = coords[1] / 1000 * height
                    x2 = coords[2] / 1000 * width
                    y2 = coords[3] / 1000 * height
                    
                    color = colors[i % len(colors)]
                    
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Draw label background and text
                    # Calculate text size
                    left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
                    text_w = right - left
                    text_h = bottom - top
                    
                    # Draw background for text to make it readable
                    # Position label above the box if possible, otherwise inside/below
                    label_y = y1 - text_h - 4
                    if label_y < 0: label_y = y1
                    
                    draw.rectangle([x1, label_y, x1 + text_w + 4, label_y + text_h + 4], fill=color)
                    draw.text((x1 + 2, label_y), label, fill="white", font=font)
            except ValueError:
                continue

        output_filename = f"visualized_{prompt_name.lower().replace(' ', '_')}.png"
        img.save(output_filename)
        print(f"[+] Visualized image saved to: {os.path.abspath(output_filename)}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")

def main():
    path = get_image_path()
    
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    prompts = {
        "1": {"name": "Layout Analysis", "prompt": "<|grounding|>Given the layout of the image.", "ext": "txt"},
        "2": {"name": "Free OCR", "prompt": "Free OCR.", "ext": "txt"},
        "3": {"name": "Parse Figure", "prompt": "Parse the figure.", "ext": "md"},
        "4": {"name": "Extract Text", "prompt": "Extract the text in the image.", "ext": "txt"},
        "5": {"name": "Markdown Conversion", "prompt": "<|grounding|>Convert the document to markdown.", "ext": "md"}
    }

    while True:
        print("\n--- DeepSeek-OCR Menu ---")
        print(f"Image: {path}")
        for key, val in prompts.items():
            print(f"{key}. {val['name']}")
        print("q. Quit")
        
        choice = input("\nSelect a prompt (1-5) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            break
            
        if choice in prompts:
            selected = prompts[choice]
            print(f"\n--- Running: {selected['name']} ---")
            print("Processing... (this may take a moment)")
            
            try:
                response = chat(
                    model='deepseek-ocr:latest',
                    messages=[
                        {
                            'role': 'user',
                            'content': selected['prompt'],
                            'images': [path],
                        }
                    ],
                )
                content = response.message.content
                print("\n--- Output ---")
                print(content)
                print("-" * 20)
                
                # Save to file
                save_output(content, selected['name'], selected['ext'])
                
                # Visualization for grounding prompts
                if "<|grounding|>" in selected['prompt']:
                    visualize_grounding(path, content, selected['name'])
                    
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()