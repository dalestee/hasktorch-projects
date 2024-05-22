module Rnn (rnn) where

import Torch.Layer.RNN(RnnHypParams(..))
import Torch.Tensor
import Torch.Functional
import Torch.Device

rnn :: IO ()
rnn = do
    putStrLn "Rnn"

    where
        numEpochs = 0 :: Int
        numWords = 2000 :: Int
        wordDim = 16 :: Int

        textFilePath :: String
        textFilePath = "app/word2vec/data/review-texts.txt"
        modelPath :: String
        modelPath =  "app/word2vec/models/sample_model.pt"
        wordLstPath :: String
        wordLstPath = "app/word2vec/data/sample_wordlst.txt"

        device = Device CPU 0
        hyperParams :: RnnHypParams
        hyperParams = RnnHypParams {
            bidirectional = False, -- ^ True if BiLSTM, False otherwise
            inputSize = numWords, -- ^ The number of expected features in the input x
            hiddenSize = numWords, -- ^ The number of features in the hidden state h
            numLayers = 1, -- ^ Number of recurrent layers
            hasBias = False -- ^ If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        }

        -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
        itr = 0 :: Int
        beta1 = 0.9 :: Float
        beta2 = 0.999 :: Float
        lr = 1e-1 :: Tensor
        dim = Dim 0 :: Dim