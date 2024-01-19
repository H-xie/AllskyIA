"""
star matching based on photo coordinates
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


class StarMatcher:
    """
    StarMatcher class

    Attributes:
        catalog: catalog of stars in Alt/Az
        sources: extracted stars

    """

    def __init__(self, distance_upper_bound=5):
        """
        catalog must have columns 'calculated_x' and 'calculated_y'
        sources must have columns 'extracted_x' and 'extracted_y'

        assertion included.
        Args:
        """
        self.distance_upper_bound = distance_upper_bound

    def match(self, catalog: pd.DataFrame, sources: pd.DataFrame):
        """
        match stars.
        Find the nearest star in catalog for each extracted star.

        if multiple extracted stars match the same catalog star, only the nearest one is kept.
        Returns:
            A pd.DataFrame of matched stars.[(all columns of stars_altaz), (all columns of sources)]
        """
        self.prepare(catalog, sources)

        catalog_tree = KDTree(catalog[['calculated_x', 'calculated_y']].values)
        extracted_tree = KDTree(sources[['extracted_x', 'extracted_y']].values)

        # match stars
        distance, extracted_id = catalog_tree.query(extracted_tree.data, 1,
                                                    distance_upper_bound=self.distance_upper_bound)

        # ensure the match is one-to-one
        index_dict = dict()
        for i, cat_id in enumerate(extracted_id):
            if cat_id < len(catalog):
                if cat_id not in index_dict:
                    index_dict[cat_id] = dict()
                index_dict[cat_id][i] = distance[i]

        for d in index_dict:
            if len(index_dict[d]) > 1:
                min_value = min(index_dict[d].values())
                for i in index_dict[d]:
                    if index_dict[d][i] != min_value:
                        extracted_id[i] = len(catalog)
                        distance[i] = np.inf

        sources['match_index'] = extracted_id
        sources = sources[sources.match_index != len(catalog)].reset_index(drop=True)

        # merge stars_altaz and sources
        result = catalog.merge(sources, left_index=True, right_on='match_index', how='left')
        result.set_index('match_index', drop=True, inplace=True)
        return result

    def prepare(self, catalog: pd.DataFrame, sources: pd.DataFrame):
        """
        prepare for matching
        Returns:
            None
        """

        # assert stars_altaz and sources are not None
        assert catalog is not None
        assert sources is not None

        # assert sources has columns 'calculated_x' and 'calculated_y'
        assert 'calculated_x' in catalog.columns
        assert 'calculated_y' in catalog.columns

        # assert sources has columns 'extracted_x' and 'extracted_y'
        assert 'extracted_x' in sources.columns
        assert 'extracted_y' in sources.columns
