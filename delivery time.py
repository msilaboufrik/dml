import pandas as pd
import numpy as np 
import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime
data= pd.read_csv(r'/home/r/Downloads/deliverytime.csv')
#set the earth's radius (in kilometers)
r=6371
#convert degrees to radians 
def deg_to_rad(degrees):
    return degrees*np.pi /180
# function to calculate the distance between two points using the haversine formula 
def dist_calculate(lat1,lon1,lat2,lon2):
    d_lat=deg_to_rad(lat2-lat1)
    d_lon=deg_to_rad(lon2-lon1)
    a=np.sin(d_lat/2)**2+np.cos(deg_to_rad(lat1))*np.cos(deg_to_rad(lat2))*np.sin(d_lon/2)**2
    c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return r*c
#Calculate the distance between each pair of points
data['Distance']=np.nan
for i in range(len(data)):
    data.loc[i,'Distance']=dist_calculate(data.loc[i, 'Restaurant_latitude'], 
                                        data.loc[i, 'Restaurant_longitude'], 
                                        data.loc[i, 'Delivery_location_latitude'], 
                                        data.loc[i, 'Delivery_location_longitude'])
fig=px.scatter(data,
                x='Distance',
                y='Time_taken(min)',
                size='Time_taken(min)',
                #trendline="ols",
                title='Relationship between distance and Time taken'
)
fig.show()
fig1=px.scatter(data,
               x='Delivery_person_Age',
               y='Time_taken(min)',
               size='Time_taken(min)',
               color='Distance',
               #trendline='ols',
               title='relationship between time taken and age ')
fig1.show()
figure = px.scatter(data, 
                    x="Delivery_person_Ratings",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "Distance",
                    #trendline="ols", 
                    title = "Relationship Between Time Taken and Ratings")
figure.show()
fig3 = px.box(data, 
             x="Type_of_vehicle",
             y="Time_taken(min)", 
             color="Type_of_order")
fig3.show()
#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["Delivery_person_Age", 
                   "Delivery_person_Ratings", 
                   "Distance"]])
y = np.array(data[["Time_taken(min)"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)
# creating the LSTM neural network model
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()
print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])
print("Predicted Delivery Time in Minutes = ", model.predict(features))
print(data.head(5))