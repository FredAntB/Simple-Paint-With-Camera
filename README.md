# Simple Paint

This is a mediapipe project to allow the user to draw on the screen without touching it.

The program itselfs needs access to the camera, so that the program can track the fingers location and act accordingly for the drawing tools.

There are currently four drawing tools:

- A brush -> Actually works more as a marker due to it's thickness
- A pencil -> Has lower thickness than the brush
- An eraser
- A figure -> currently only line, rectangles and circles are supported

There are two pickers, a color one used for the brush and pencil, and a figure one that allows the user to choose between the line, rectangles an circles.

Note: Currently the figures are being constantly drawn, might be a future feature to be able to chose a proper placement for the figure without it being drawn again and again.

Use:

- Once you have the software running with camera access, you have to use your right hand.
- Lift up your index and middle finger, this will put you on Selection Mode, and display a cursor between your fingers.
  - Once on selection mode, you will be able to chose between the available drawing tools, for this purpose move your fingers until you see them going behind the desired tool.
- Lift up only your index finger to enter on Drawing Mode, if you are using a brush, a pencil, or an eraser you will be able to draw over the displayed view, or erase the drawn content.
- By using your thumb and index finger you enter on Figure Mode, once here, you widen or close the distance between both fingers to give the desired size to your figure (currently under development for better user experience).

Note: Whatever mode you end up using, you will see a text displayed on the screen at the right side mentioning the mode in use.
