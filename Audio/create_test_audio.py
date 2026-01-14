"""
Creates sample audio files for testing
pip install edge-tts
"""

import asyncio
import edge_tts
import os

# ═══════════════════════════════════════════════════════════════════════════════
# 📁 SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = "test_audios"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Turkish voices
VOICE = "tr-TR-EmelNeural"  # Female (clearer)
# VOICE = "tr-TR-AhmetNeural"  # Male

# ═══════════════════════════════════════════════════════════════════════════════
# 📝 TEST SENTENCES
# ═══════════════════════════════════════════════════════════════════════════════
test_sentences = [
    ("test_01_python_nedir.mp3", "Python nedir?"),
    ("test_02_hesaplama.mp3", "Yüz yirmi beş çarpı kırk sekiz kaç eder?"),
    ("test_03_saat_kac.mp3", "Şu an saat kaç?"),
    ("test_04_yapay_zeka.mp3", "Yapay zeka nedir?"),
    ("test_05_baskent.mp3", "Türkiye'nin başkenti neresidir?"),
    ("test_06_enflasyon.mp3", "Enflasyon ne demek?"),
    ("test_07_hava_durumu.mp3", "İstanbul'da bugün hava nasıl?"),
    ("test_08_uzun_karmasik.mp3", 
     "Yapay zeka ve makine öğrenmesi arasındaki temel farklar nelerdir? "
     "Günümüzde en çok hangi sektörlerde kullanılmaktadır?"),
    ("test_09_coklu_hesap.mp3", "Bin iki yüz otuz dört çarpı elli altı kaç eder?"),
    ("test_10_belirsiz.mp3", "Şey, hani şu internetten arama yapan program var ya, onu sor bana."),
    ("test_11_hizli.mp3", "İstanbul, Ankara, İzmir, Bursa ve Antalya şehirlerinin nüfusları kaçtır?"),
    ("test_12_teknik.mp3", "Kubernetes ile Docker arasındaki fark nedir?"),
    ("test_13_guncel.mp3", "Türkiye'de bugünkü dolar kuru ne kadar?"),
    ("test_14_felsefe.mp3", "Yapay zeka bilinç kazanabilir mi?"),
    ("test_15_stres.mp3",
     "Merhaba, ben bugün çok yorgunum. Sabah erkenden kalktım, işe gittim, "
     "toplantılara katıldım, öğle yemeğinde salata yedim, akşam eve döndüm. "
     "Şimdi sana bir soru sormak istiyorum. "
     "Yarın İstanbul'da hava nasıl olacak? Hafta sonu piknik yapabilir miyim?"),
    ("test_16_selamlama.mp3", "Merhaba, nasılsın? Bugün sana birkaç sorum olacak."),
    ("test_17_tesekkur.mp3", "Çok teşekkür ederim, bu bilgi çok faydalı oldu."),
    ("test_18_hava_ankara.mp3", "Ankara'da hava durumu nasıl? Yarın yağmur yağacak mı?"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 🔊 AUDIO GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

async def create_audio(filename: str, text: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    communicate = edge_tts.Communicate(text=text, voice=VOICE)
    await communicate.save(filepath)
    return filepath

async def main():
    print(f"🎙️ Creating audio files... ({VOICE})")
    print(f"📁 Folder: {OUTPUT_DIR}/")
    print("-" * 50)
    
    for filename, text in test_sentences:
        try:
            await create_audio(filename, text)
            print(f"✅ {filename}")
        except Exception as e:
            print(f"❌ {filename}: {e}")
    
    print("-" * 50)
    print(f"🎉 {len(test_sentences)} audio files created!")

if __name__ == "__main__":
    asyncio.run(main())
