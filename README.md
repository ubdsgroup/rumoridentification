# Rumor Identification Using Random Walks
 
For this code file, given a known rumor, we are looking for the unknown tweets that have high relevance to the known rumor using random walk method. We are trying to find the stationary probability of all the other tweets if we start from a rumor tweet to do the random walk. Tweets with high stationary probability means that it has higher chance to be landed in random walk, also means it is highly relevant to the rumor tweet. The stationary probability will be presented as a vector with length of the number of tweets. By updating the vector using adjacency matrix till it converges, we can get the stationary probability.

