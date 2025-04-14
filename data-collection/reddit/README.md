# reddit
The Reddit API! Investigated by Matthew Dong.
Some questions that we should be asking ourselves:
Which posts do we deem the most valuable?
- Do we just get the 1000 most recent posts of today? I think splitting by time should be possible because you can get the time in UTC when a post was created.
- Do we get the 1000 most highly rated posts of today?
- How deep in comment chains do we want to go? How does that even work? Maybe we should go to a depth of 3.
- Maybe if we get the topic of the post we can determine if it's relevant or not?
- To find all available flairs for a subreddit we can just make a post draft. Flairs might be useful for determining post topic.
- The API can indicate if a post was edited.
- Cross posting across subreddits?
- High number of comments = high engagement
- Controversial flag?
- Get awards provided to a post or comment. Let's ignore Low Quality Article.
- Did Reddit truly remove downvotes?
- "Free API access rates are as follows: 100 queries per minute per OAuth client id if you are using OAuth authentication 10 queries per minute if you are not using OAuth authentication"

As a side note, I didn't know USERNAME was already a system environment variable! The more you know.
