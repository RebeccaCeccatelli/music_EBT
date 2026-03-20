import sys
import torch
import os

def run_diagnostics():
    print("--- 🛠️ Music_EBT Final Environment Check ---")
    
    # Debugging the physical path
    repo_path = os.path.join(os.getcwd(), "data/mus/symbolic/tokenization/anticipation")
    print(f"📂 Checking folder: {repo_path}")
    
    if os.path.exists(repo_path):
        print(f"📄 Folder contents: {os.listdir(repo_path)}")
    else:
        print("❌ ERROR: The anticipation folder does not exist!")

    try:
        # Based on your discovery: tokenization -> anticipation (submodule) -> anticipation (package)
        from tokenization.anticipation.anticipation import config
        import tokenization.anticipation.anticipation as anticipation
        
        print(f"✅ Import Success!")
        print(f"📍 Package Location: {anticipation.__file__}")
        print(f"🎼 MIDI Resolution: {config.TIME_RESOLUTION} bins/sec")
        
    except ImportError as e:
        print(f"❌ Import Failed: {e}")
        print(f"🔍 Current sys.path: {sys.path[-3:]}")
        sys.exit(1)

    # Check for GPU
    cuda = torch.cuda.is_available()
    print(f"🎮 GPU Available: {'✅ Yes' if cuda else '❌ No (Standard for Login Node)'}")
    
    print("\n🚀 Environment is officially READY.")

if __name__ == "__main__":
    run_diagnostics()