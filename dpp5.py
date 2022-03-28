from numpy import nan
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.impute import RobustImputer
from sagemaker_sklearn_extension.preprocessing import NALabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sagemaker_sklearn_extension.preprocessing import ThresholdOneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(
    column_names=[
        'Your es\xadtim\xadated rev\xaden\xadue (USD)', '', 'Video Length',
        'USA%_veiws', 'IN%_views', 'Other_country%_ views',
        'Com\xadments ad\xadded', 'Shares',
        'Av\xader\xadage per\xadcent\xadage viewed (%)', 'Views',
        'Watch time (hours)', 'Sub\xadscribers', 'Im\xadpres\xadsions',
        'Im\xadpres\xadsions click-through rate (%)', 'Monday', 'Saturday',
        'Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'DateScience?',
        'Average_view_duration_(s)', 'Like_Dislike_Ratio',
        'Reply_Comment_Count', 'Like_Comment_Count'
    ],
    target_column_name='Your es\xadtim\xadated rev\xaden\xadue (USD)'
)


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features can be parsed as numeric.

    numeric = HEADER.as_feature_indices(
        [
            '', 'Video Length', 'USA%_veiws', 'IN%_views',
            'Other_country%_ views', 'Com\xadments ad\xadded', 'Shares',
            'Av\xader\xadage per\xadcent\xadage viewed (%)', 'Views',
            'Watch time (hours)', 'Sub\xadscribers', 'Im\xadpres\xadsions',
            'Im\xadpres\xadsions click-through rate (%)', 'Monday', 'Saturday',
            'Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'DateScience?',
            'Average_view_duration_(s)', 'Like_Dislike_Ratio',
            'Reply_Comment_Count', 'Like_Comment_Count'
        ]
    )

    # These features contain a relatively small number of unique items.

    categorical = HEADER.as_feature_indices(
        [
            'Com\xadments ad\xadded', 'Monday', 'Saturday', 'Sunday',
            'Thursday', 'Tuesday', 'Wednesday', 'DateScience?',
            'Reply_Comment_Count', 'Like_Comment_Count'
        ]
    )

    numeric_processors = Pipeline(
        steps=[
            (
                'robustimputer',
                RobustImputer(strategy='constant', fill_values=nan)
            )
        ]
    )

    categorical_processors = Pipeline(
        steps=[('thresholdonehotencoder', ThresholdOneHotEncoder(threshold=8))]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric_processing', numeric_processors, numeric
            ), ('categorical_processing', categorical_processors, categorical)
        ]
    )

    return Pipeline(
        steps=[
            ('column_transformer', column_transformer
            ), ('robuststandardscaler', RobustStandardScaler())
        ]
    )


def build_label_transform():
    """Returns the model definition representing feature processing."""

    return NALabelEncoder()
