import pandas as pd
import os

def dataprep_task2(path):
    """Dataprep for Task2 It will return the new data
    :param path: Path to the article's taks3 labels file.

    Example:
    >>> dataprep_task2("datasets-v5/tasks-2-3/train/article111111112.task2.labels")
    Note the method will return Pandas DataFrame
    """
    dir_name = os.path.dirname(path)
    article_id = os.path.basename(path).split('.')[0]
    article_name = os.path.join(dir_name, f'{article_id}.txt')

    with open(article_name, 'r') as f:
        records = f.readlines()

    df = pd.DataFrame(records, columns=['sentences'])

    another_df = pd.read_csv(path, sep='\t', names = ['article', 'N_sentence', 'is_propaganda'])

    result_df = pd.concat([df, another_df], axis=1)

    return result_df.loc[result_df['sentences'] != '\n', :]


def dataprep_task3(path):
    """Dataprep for Task3 It will return the new data
    :param path: Path to the article's taks3 labels file.

    Example:
    >>> dataprep_task2("datasets-v5/tasks-2-3/train/article111111112.task3.labels")
    Note the method will return Pandas DataFrame
    """
    dir_name = os.path.dirname(path)
    article_id = os.path.basename(path).split('.')[0]
    article_name = os.path.join(dir_name, f'{article_id}.txt')

    with open(article_name, 'r') as f:
        article = f.read()

    df = pd.read_csv(path, sep='\t',
                     names = ['article_id', 'propaganda_technique', 'fragment_start', 'fragment_end'])

    df['fragment_phrase'] = df.apply(lambda x: article[x['fragment_start']: x['fragment_end']], axis=1)

    return df
