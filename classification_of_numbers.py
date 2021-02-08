import pygame
import keyboard as kb
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('model.h5')
model.summary()

pygame.init()
screen = pygame.display.set_mode((128*3,128*3))
screen.fill((0, 0, 0))
pygame.display.set_caption("NNPaint")
clock = pygame.time.Clock()
draw_on = False
radius = 20
is_predict = False

while True:
    e = pygame.event.wait()
    if e.type == pygame.QUIT:
        exit(0)
    if e.type == pygame.MOUSEBUTTONDOWN:
        pygame.draw.circle( screen, (255,255,255), e.pos,radius)
        draw_on = True
    if e.type == pygame.MOUSEBUTTONUP:
        draw_on = False
    if e.type == pygame.MOUSEMOTION:
        if draw_on:
            pygame.draw.circle( screen, (255,255,255), e.pos, radius )
            is_predict = False
        last_pos = e.pos

    if kb.is_pressed("esc"):
        screen.fill( (0, 0, 0) )
    if not draw_on and not is_predict:
        pygame.image.save(screen,"temp.png")
        load_img = cv2.imread( 'temp.png' )
        img = cv2.resize( load_img, (28, 28) )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #plt.figure( figsize=(5, 5))
        #plt.imshow(img)
        img = (np.expand_dims( img, 0 ))
        prediction = model.predict(img)
        print(prediction)
        val = 0
        for i,row in enumerate(prediction[0]):
            if row > val <= 1:
                val = row
                pygame.display.set_caption(f"Это цифра: {i}")
                break
        is_predict = True
    pygame.display.flip()