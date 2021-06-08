# OffNote Labs Research Task

Hi. This is the my implementation of the Vision Task for Offnote Labs.

## Overview

The aim of the project was to find all instances of text embedded in a regular video. This is a popular research problem in Computer Vision, commonly known as Optical Character Recognition (OCR). For this task, I used a video describing [Apple's event in October 2020](https://www.youtube.com/watch?v=Gz8vBoEFArA), since it had a healthy amount of text I could extract.

## Project Structure

I used the following pipeline for this task : 
![image](https://user-images.githubusercontent.com/71698670/121188813-1a298180-c887-11eb-89d5-a5cdc182887f.png)

The first step to split the input video into frames. I used OpenCV’s [EAST text detector](https://arxiv.org/abs/1704.03155) to detect the presence of text in an image. The EAST text detector gives us the bounding box (x, y)-coordinates of text ROIs.

We extract each of these ROIs and then pass them into Tesseract v4’s LSTM deep learning text recognition algorithm. 
Tesseract is a very popular OCR engine that has an inbuilt LSTM model for text recognition. The output of the LSTM gives us our actual OCR results.
Finally, we draw the OpenCV OCR results on our output frame and display the results.

## Results

Our OCR engine works fairly well. Here is a snapshot from our text detector : 

https://user-images.githubusercontent.com/71698670/121201611-47c7f800-c892-11eb-832d-dcebb797329d.mp4

And here's the corresponding output from our text recognition system :

![image](https://user-images.githubusercontent.com/71698670/121202559-04ba5480-c893-11eb-87f6-a2e17745255e.png)

### A few observations I made while tuning this model :

**Padding :** A small amount of padding works well for most models. However, as we increase the padding, there may be overlapping between different words in our image/video, leading to poor results from our text recognition engine.

**Frames per second:** The video that I used ran at about 30 FPS. I reduced that to 10 FPS, in order to generate quicker results during test time.

## Concluding Notes :

Our OCR engine is far from perfect, and it does output a few weird words from time to time :

![image](https://user-images.githubusercontent.com/71698670/121206336-ffaad480-c895-11eb-8da6-e565b4c83abf.png)

I loved working on this project, and will be looking for ways to improve it in the future. 
