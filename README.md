# nb-combination
ensemble classifier with naive bayes combination

The algorithm is implemented in `nb_comb.py`

## use
just run `script.py`

## Example

```python
X=np.array([[1,2,3.1],[2,2,2.3],[1,1,1.1], [1,1,2.1]])
Y= [0,1,0,1]
from sklearn.naive_bayes import *
estimators = [('multinomial', MultinomialNB()), ('gauss', GaussianNB())]
model = NBAdditive(estimators)
model.fit(X, Y, inds=[[0,1], [2]])
print(model.predict(X))
```
