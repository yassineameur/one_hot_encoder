
class Encoder:

    def __init__(self, prefix_sep='_', drop_first=False, dummy_na=False):

        """
        Convert categorical variables into dummy/indicator variables

        Parameters
        ----------

        prefix_sep : string, default '_'
            If appending prefix, separator/delimiter to use.
        dummy_na : bool, default False
            Add a column to indicate NaNs, if False NaNs are ignored.
        drop_first : bool, default False
            Whether to get k-1 dummies out of k categorical levels by removing the
            first level.

        Examples
        --------
        >>> import pandas as pd
        >>> s = pd.Series(list('abca'))

        >>> pd.get_dummies(s)
           a  b  c
        0  1  0  0
        1  0  1  0
        2  0  0  1
        3  1  0  0

        >>> s1 = ['a', 'b', np.nan]

        >>> pd.get_dummies(s1)
           a  b
        0  1  0
        1  0  1
        2  0  0

        >>> pd.get_dummies(s1, dummy_na=True)
           a  b  NaN
        0  1  0    0
        1  0  1    0
        2  0  0    1

        >>> df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
        ...                    'C': [1, 2, 3]})

        >>> pd.get_dummies(df, prefix=['col1', 'col2'])
           C  col1_a  col1_b  col2_a  col2_b  col2_c
        0  1       1       0       0       1       0
        1  2       0       1       1       0       0
        2  3       1       0       0       0       1

        >>> pd.get_dummies(pd.Series(list('abcaa')))
           a  b  c
        0  1  0  0
        1  0  1  0
        2  0  0  1
        3  1  0  0
        4  1  0  0

        >>> pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)
           b  c
        0  0  0
        1  1  0
        2  0  1
        3  0  0
        4  0  0

        See Also
        --------
        Series.str.get_dummies
        """

        self.prefix_sep = prefix_sep
        self.drop_first = drop_first
        self.dummy_na = dummy_na

        self.categories_by_column = {}
        self.columns = None

    def _get_columns_to_encode(self, data, columns):
        """
        :param data: pandas Dataframe.
        :param columns: The list of columns to encode.
        """
        if columns:
            self.columns = columns
            return
        self.columns = list(data.select_dtypes('object').columns)

    def _get_categories_for_column(self, data, column):
        """
        :param data: pandas Dataframe.
        :param column: The column name
        :return: the list of categoies for the specified column.
        """
        if self.dummy_na:
            categories = data[column].unique()
        else:
            categories = data[column][data[column].notnull()].unique()

        if self.drop_first and len(categories) > 1:
            categories = categories[1:]
        return categories

    def _get_categories_by_column(self, data):
        """
        :param data: panda Dataframe.
        :return: a dict that contains categories by column.
        """

        self.categories_by_column = {
            column: self._get_categories_for_column(data, column) for column in self.columns
            }

    def _get_category_column_name(self, category, column, columns_list):
        """
        :param category: The category name.
        :param column: The category column
        :param columns_list: The columns list
        :return: a new column name for the category with making sure that you will never have duplicated
        columns names.
        """
        category_column_name = '{}{}{}'.format(column, self.prefix_sep, str(category))
        if category_column_name not in columns_list:
            return category_column_name

        index = 0
        while '{}{}{}'.format(category_column_name, self.prefix_sep, index) in columns_list:
            index += 1
        return '{}{}{}'.format(category_column_name, self.prefix_sep, index)

    def fit(self, data, columns=None):
        """
        This function is used to inspect the reference dataframe.
        It will fetch for the columns to convert and the categories by column.
        :param data: pandas DataFrame
        :param columns: list-like, default None
            Column names in the DataFrame to be encoded.
            If `columns` is None then all the columns with
            `object` or `category` dtype will be converted.
        """

        self._get_columns_to_encode(data, columns)
        self._get_categories_by_column(data)

    def get_dummies(self, data, edit=False):
        """
        :param data: a pandas dataframe
        :param edit: A boolean indicating whether you want to edit your data or not.
        If edit = True, your data will be modified: it will be the result. ELse, your
        data will not be touched. We recommend to set edit to False if you have sufficient memory.
        :return: A dataframe with dummy columns.
        """
        if not edit:
            data_to_edit = data.copy(deep=True)
        else:
            data_to_edit = data

        data_columns = list(data_to_edit.columns)

        for column in self.columns:
            categories = self.categories_by_column[column]
            for category in categories:
                category_column_name = self._get_category_column_name(category, column, data_columns)
                data_to_edit[category_column_name] = data_to_edit[column].map(
                    lambda x: 1 if str(x) == str(category) else 0)

            del data_to_edit[column]

        return data_to_edit
