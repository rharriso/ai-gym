May 12, 2018

* Fixed preprocessor (no difference in learning)
* Fixed action repetition (no difference in learning, training is much faster)
    * s' is the state after repitition
* added training loss graph

May 14, 2018

Added clipping of -0.5 - 0.5 to optimizer (no difference in learning)

May 15, 2018

Need to try adding momentum via wrapping tf.training.RMSPropOptimizer (has momentum arg)
* https://keras.io/optimizers/#tfoptimizer
* https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
