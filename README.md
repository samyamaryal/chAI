# UXIQ (UX Intelligence on Edge)
Track, analyze, and summarize user interactions (mouse, keyboard, facial expressions) entirely offline. Powered by Snapdragon edge AI for secure, local UX evaluation without internet dependency. Ideal for UX research and accessibility testing.

Recommend to Setup Anaconda environment

# If not working try... (THIS WORKS)

# Step 1: Clone and install locally (recommended)
git clone https://github.com/quic/ai-hub-models.git
cd .\ai-hub-models\
pip install -e . 
pip install qai_hub_models
pip install "qai_hub_models[yolov7]"
pip install "qai-hub-models[facemap-3dmm]"

conda init
conda activate uxtracker

# Step 2: Clean conflicting OpenCV and numpy versions
pip uninstall opencv-python-headless opencv-python opencv-contrib-python numpy -y

# Step 3: Install correct dependencies
pip install numpy==1.24.4
pip install opencv-contrib-python==4.8.0.74

# Validate installations
pip show numpy opencv-contrib-python qai-hub onnxruntime
pip list | findstr opencv

# Optional 
pip install --upgrade pip setuptools wheel
pip install packaging==23.2   


