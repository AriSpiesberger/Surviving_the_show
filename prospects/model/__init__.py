"""Model training and inference.

Nothing is re-exported here on purpose: the members of this package are
either standalone entry points or heavyweight modules (xgboost, sklearn),
and eagerly importing them would make `import prospects` slow and fragile.
Import the specific module you need.
"""
