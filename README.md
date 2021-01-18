# HR Interview assignment

Complete the assignment in Python using any libraries you like.

1. Do a manual ad-hoc analysis: 
	1. Find a department with more than 20 workers with biggest termination rate (over the whole time period). Termination rate is defined as `(# of terminated workers) / (# of workers)`.
	2. Which Termination reason contributed the most to the high termination rate?
	3. Is the distribution of Termination reasons similar in other departments?
2. You are provided with unfinished model that predicts who is going to leave the company in `model.py`. Fill in two missing methods:
	1. `split_train_valid_data`
	2. `cross_validate_rfc_model`
3. Train the model on data from 2014, using just (included already in `prepare_data_columns`)
		* numerical features,
		* gender,
		* indicator if they are from `Store Management` department,
		* indicator if they are from `Produce` department.
4. Predict and evaluate the model for the year 2015.

We may ask during the interview: which parts of the model you would improve or refactor?
