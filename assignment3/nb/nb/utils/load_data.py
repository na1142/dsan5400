from pathlib import Path
import pandas as pd


class DataLoader:

    def __init__(self, folder):
        """
        :param folder: a path to a directory containing the kennedy, johnson
                       and unlabeled subdirectories
        """
        self.path = Path(folder)
        self.author_to_id_map = {'kennedy': 0, 'johnson': 1}

    def _make_df_from_dir(self, dir_name, df):
        """
        Takes as input a directory name to construct a df from and returns
        the updated df
        :param dir_name: name of a subdirectory of the data folder
        :param df: a pandas DataFrame to add rows to
        :return: updated pandas DataFrame
        """
        for f in self.path.glob(f'./{dir_name}/*.txt'):
            with open(f, encoding='utf-8') as fp:
                text = fp.read()
            if dir_name in ('kennedy', 'johnson'):
                author = dir_name
            else:
                author = f.stem.split('_')[-1]
            temp_df = pd.DataFrame({'author': [author], 'text': [text]})
            df = pd.concat([df, temp_df], ignore_index=True)
        return df

    def build_dataframe(self):
        """
        Returns two DataFrames storing the text from the files, one for the
        training data and one for the test data (unlabeled)
        :return: a tuple of pandas DataFrames
        """
        df_train = pd.DataFrame(columns=['author'])
        df_test = pd.DataFrame(columns=['author'])

        for p in self.path.iterdir():
            if p.name in ('kennedy', 'johnson'):
                df_train = self._make_df_from_dir(p.name, df_train)
            elif p.name == 'unlabeled':
                df_test = self._make_df_from_dir(p.name, df_test)

        # replace the strings for the author names with numeric codes (0, 1)
        df_train['author'] = df_train['author'].map(self.author_to_id_map)
        df_test['author'] = df_test['author'].map(self.author_to_id_map)
        return df_train, df_test