#Basic linear model
import tensorflow as tf
x= [1.0,3.0,1.5,5.0,0.5,2.0,6.0]
y=[3.0,7.0, 3.0, 11.0, 2.0,5.0,13.0]

model= tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="sgd", loss="mse")
model.fit(x,y, epochs=15, verbose=1)

print(model.predict([7.0]))