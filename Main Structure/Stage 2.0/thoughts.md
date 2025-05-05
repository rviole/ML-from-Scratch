So far, it is the first time i developed a non neural network for a non-linear problem. It opened a lot of new things to me.
1. Chain rule - it is insane how i was trying to find work arounds just not to use this rule that i didn't understand. After this project, the chain rule seems to me as one of the most beatiful concepts not just of math, but of life. It is genious. It literally carries information over the whole sequence of changes. 
    
    - it feels like a domino: You have domino 1, 2, and 3. It calculated how domino 3 will land based on how domino 2 will land based on how domino 1 will land. It is like accounting for sesitivity of each change. Absolutely stunning.

2. I also discovered the insights of ReLU activation funciton. It seemed to me so simple, but it does such a crazy job. It just literally breaks a line at one point, but introduces non-linearity. Without its simple nature, out whole neural network is just one big linear transformation.
    
    - I discovered that if i we have an input and 1 relu function for the output, our straight line will break at one point, one of the sub-lines will always be on x-axis (y=0) and another will be a simple line that we can tune.
    