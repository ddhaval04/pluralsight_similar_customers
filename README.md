# Pluralsight Customer Similarity

## Docker

Please make sure [Docker](https://docs.docker.com/) is installed on your computer.

Then run, `docker-compose up`. This will then spin up docker containers that 1) runs tests, 2) reads the data, 3) processess the data, 4) performs SVD on the processed data, 5) fits KNN on the customer (user) embeddings and pickles the model and the metadata in a trained_model`directory and 6) finally starts the flask service.

To test, make a request to `http://0.0.0.0:5000/predict`with a json string `{'user_handle': 1}`. This will give you a summary of the similar customers.

## Questions:

1) Tell us about your similarity calculation and why you chose it.
In this challenge, I have considered an approach of a low-rank matrix factorization and nearest-neighbors search to compute the customer similarity. The high-level idea here is that, once I build a user-course_tag interaction matrix (here, the interaction is the average time a user spends on the course_tag defined by the `view_time_seconds` feature), we can decompose this matrix into a "user" features matrix, singular values and "course_tag" features.
Once I have these user embeddings, I perform a nearest neighbor search for a query embedding and generate summary of similar customers.

2) We have provided you with a relatively small sample of users. At true scale, the
number of users, and their associated behavior, would be much larger. What
considerations would you make to accommodate that?
The current approach has 2 major bottlenecks, 1) At true scale, performing matrix factorization on a large dataset poses a lot of challenges, ex: fitting data into memory. Hence it probably isn't a scalable solution for measuring customer similarity. Probably considering a representational learning approach might be helpful. 2) Second challenge is the retrieval process, assuming you have the customer embeddings generated, performing knn search is a costly operation as it strives for an exact match solution. At scale, we would have to make a trade-off between speed and accuracy, and hence building indexes for the customer embeddings using an approximate nearest neighbor approach helps. One more issue with these indexes is that you have to load them in memory, which is not always possible and hence possibly sharding the indexes across multiple nodes can help improve scalability.

3) Given the context for which you might assume an API like this would be used, is
there anything anything else you would think about? (e.g. other data you would
like to collect)
The current approach is a naive way to identify similar customers (users). Having demographic information about users, will be useful in a way where it can help us segment users into specific cohorts and find patterns in their preferences about the course. Also, having raw content (be it lecture notes/videos/slides etc.) about the course available to us can help us determine what exactly a user is interested in.
