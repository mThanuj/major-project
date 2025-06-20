# MFCC-Guided Convolutional Autoencoders: A Hybrid Framework for Noise-Resilient, Edge-Optimized Keyword Spotting

## Directories

### express-backend

- This directory contains the backend for the express app.
- It is used as the base for our frontend to communicate with the model.
- Technologies used:
  - typescript: to handle types
  - express: to handle http routes
  - multer: to handle file uploads
  - axios: to make http requests

### flask-backend

- This directory contains the backend for the flask app.
- It's main function is to serve the model to the express backend.
- It also contains the training and evaluation code for the model.
- Technologies used:
  - flask: to handle http routes
  - torch: to handle model training

### frontend

- This directory contains the frontend for the app.
- It contains two forms:
  - One can upload a wav file.
  - The other can record a voice live.
- After the user has uploaded a wav file or has recorded a voice, the app sends a POST request to the express backend.
- Technologies used:
  - nextjs: to handle routing and rendering
  - tailwindcss: to handle styling
  - shadcn: to handle ui components
  - axios: to make http requests
  - framer-motion: to handle animations

## Installation

- Clone the repository

### express-backend-installation

- Run `npm install` in the express-backend directory.
- Run `npm dev`

### flask-backend-installation

- Run `conda env create -f environment.yml` in the flask-backend directory.
- Run `conda activate torch-cuda` in the flask-backend directory.
- Two scripts:
  - Run `python model/main.py` for training the model.
  - Run `python app.py` for serving the model to the express backend.

### frontend-installation

- Run `pnpm install` in the frontend directory.
- Run `pnpm dev`
