# DQN
A standard DQN implementation

### **Note for Intel-based Mac users**
You can safely ignore the message for Mac M1 users, as TensorFlow is fully-compatible on Intel systems, as it always has been. The same applies to all of the other packages, such as gym. Nothing unusual will have to be performed to get a successful installation.

### **Note for Mac M1 users**
**This project is compatible with Mac M1, despite the mess that faulty mess that is Mac M1 package compatibility at the moment**. While most computer-vision and data science modules, such as CV2, are highly difficult if not impossible to install with TensorFlow and the like at the moment, this project resorts to libraries that are functional on both Intel-based and Mac M1 systems. Apple has released a mac M1-optimized TensorFlow build which is still in alpha as of February 4, 2021. Until there are more stable releases of these packages for the M1 system, you should follow the instructions to install [`tensorflow_macos`](https://github.com/apple/tensorflow_macos) on Python 3.8.6. Then, install OpenAI Gym and the remaining packes via the `requirements.txt` file. Make sure that everything is installed under Python version 3.8.6. You will likely get a rendering issue on the first attempt at installation due to some issues with Pyglet rendering for gym and its compatibility in MacOS Big Sur software and Mac M1 hardware. The error message will convey that you have no display to render to, and must connect a virtual display. This is now the case; instead, you should simply install a certain version of Pyglet that is compatible at the moment: `python3.8 -m pip install pyglet==1.5.14`. At this point, you should have TensorFlow for Mac M1 and a OpenAI gym module, and all of the remaining packages used in this repo.

### **Note**
If you change the environment variable `$ENV_ID` in the `.env` file, refer to `wrapper.py`. The Breakout environment uses a different state preprocessor for reasons described in the aforementioned file. Follow the instructions given in the commented section of the file before training on a different environment.
