Approach:

I first started by doing a couple of laps, driving normally on the middle of the road.
At first I fine-tuned the VGG network.

This first method did not give satisfying results. The car was zigzagging left and right and eventually went off the road
even before the first sharp turn.

My first thought was that the data was insufficent. I drove another turn the other way around and flipped all images to
easily double my input and counterbalance the zigzagging of the car. I thought the recovery training was not that important
as long as I could keep the car in the middle of the road

This did not improve my results at all, the car had the same behavior.

As the car was only steering at + or - 25°, I thought using the keyboard to train was the issue as it has a binary output.
I started using a gamepad and redid some capture using the same method: a couple of laps driving in the middle of the road
and a couple of laps the other way around. I even trained on the second track to help the model generalize.

At that point I had around ~70k images of input, but the results weren't improving at all. Because the training was pretty
slow, I started using a way simpler network than VGG. It did not improve my results, they pretty much stayed the same but I
could iterate faster on my data.

I realized recovery training might be the solution. I recorded about 10k images of recovery from both sides of the road,
and also started using the left and right input images, correcting the steering input by +/- .25 for each of them.
The car was still zigzagging a lot but did not exit the road as fast and I managed to reach the first sharp turn.

Because the recovery training showed some improvement in my results, but not enough, I thought maybe the "regular" driving
data was too prevalent. I decided to restart my training from zero, driving only one lap in the middle of the road, and one
only doing recovery.

Once again, this did not improve my results at all. I considered my input data to be correct now, so I thought the network
was the issue.

I tried using VGG again, using more epochs. And again, no improvement, the results were even worse. The car was just driving
in a straight line, barely steering at all. At this point, I was completely at loss and almost discouraged as all my efforts
were barely making any difference.

Then came an idea I should have had since the beginning, checking the actual values my network was predicting when driving and
not only the steering angle of the car on the simulator. I saw the values were very high, instead of ranging from -1 to 1, they
were going roughly from -30 to 30. That's where I realized I was not normalizing the image I was passing to the network for prediction,
which means my testing results were completely biased since the beginning and could not yield significant learnings.

Once I started normalizing the images for prediction, the results started to actually make sense. The car was steering gently
from left to right, staying in the middle of the road most of the time. The results were still not perfect though, as the
car went out of the road just before the end of the lap.

To prevent the car from steering away from the road, I decided to increase the steering correction on the left and right images.

The model was still not good enough. It seems that it was overfitting to the track and would sometimes turn too soon. I used more data augmentation to avoid this. I added some random brightness in my images.
Also, instead of using every image I have, I started to randomize the selection.

This seemed to help avoid overfitting and the car made a whole lap without falling!
The steering angle was still changing too rapidly sometimes though.

To generate smoother data, I started going really slowly while capturing the data. This led to a way smoother variation in my angles and a way better data quality.

The car was driving way better now around the track.

Network architecture:

I buildt a small network with 6 CNNs. I put a maxpool after every 2 CNNs and one dropout after the first 4 ones.
I wanted to avoid overfitting the model.
I then added 4 FC layers. I also put two dropouts here.

