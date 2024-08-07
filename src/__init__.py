
"""
This is the code implementation for:
/// paper title ///

The implementation of generative LSTM is hugely influenced by REINVENT,
https://github.com/MarcusOlivecrona/REINVENT

For comments and bug reports, please send an email to bsbae402@gmail.com.
"""

### IMPORTANT! 
### if you don't disable the logging of the rdkit,
### you will see tons of error and warning messages when you try to MolFromSmiles().
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')