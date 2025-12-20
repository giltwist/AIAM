This was a semester project for COGS415 (Artificial & Natural Intelligence) at CWRU in the Fall 2025 Semester.

The most successful experiment for me was the basic TF-IDF method, but a core lesson from the project was that finding the right approach to machine / deep learning  is an unpredictable art.  Experimentation is mandatory, because results vary wildly across seemingly similar datasets or with small changes to parameters.  

Goldilocks of fit was an F1 score in the 85% range and was generalizable to text messages and emails.  Overfitting became obviouratefied rands at 90% F1 score, when all non-Reddit text samples were basically falsely labeled "not me."  

Stratefied random sampling based on Euclidean distance from an "average" comment could loosely be said to be about twice as effective as pure random sampling when it came to reducing the size of the training set.

Runs on Python 3.12.  Original dataset consisted of a CSV containing 1 million random Reddit comments and a second CSV containing 20,000 of my Reddit comments.

