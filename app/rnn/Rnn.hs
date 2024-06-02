{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use newtype instead of data" #-}


module Rnn (rnnMain) where

import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import GHC.Generics
import Torch.NN (Parameter, Parameterized(..), Randomizable(..))
import Torch.Serialize (loadParams)
import Torch.TensorFactories ( randnIO', zeros' )
import Torch.Autograd (makeIndependent)
import Torch.Layer.RNN      (RnnHypParams(..), RnnParams(..), rnnLayers)
import Torch.Optim          (mkAdam, foldLoop)
import Torch.Train          (update, saveParams)
import Torch.Functional     (mseLoss, Dim(..))
import Torch.Tensor         (Tensor(..), asValue, asTensor, TensorLike (asTensor), shape)
import Torch.Device         (Device(..), DeviceType (CPU))
import Torch.Layer.MLP      (ActName(Relu, Tanh))

sizeInput = 1
sizeHidden = 6
nLayers = 2

-- amazon review data
data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic)

instance FromJSON Image
instance ToJSON Image

data AmazonReview = AmazonReview {
  rating :: Float,
  title :: String,
  text :: String,
  images :: [Image],
  asin :: String,
  parent_asin :: String,
  user_id :: String,
  timestamp :: Int,
  verified_purchase :: Bool,
  helpful_vote :: Int
  } deriving (Show, Generic)

instance FromJSON AmazonReview
instance ToJSON AmazonReview

-- model
data ModelSpec = ModelSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)


data RNN = RNN {
  rnnParams :: RnnParams
} deriving (Show, Generic, Parameterized)

data Model = Model {
  emb :: Embedding,
  rnn :: RnnParams
} deriving (Show, Generic, Parameterized)

instance
  Randomizable
    ModelSpec
    Model
  where
    sample ModelSpec {..} =
        (Model . Embedding <$> (makeIndependent =<< randnIO' [wordDim, wordNum]))
        -- TODO: add RNN initilization
        <*> sample (RnnHypParams {
          dev = Device CPU 0,
          bidirectional = False, -- ^ True if BiLSTM, False otherwise
          inputSize = sizeInput, -- ^ The number of expected features in the input x
          hiddenSize = sizeHidden, -- ^ The number of features in the hidden state h
          numLayers = nLayers, -- ^ Number of recurrent layers
          hasBias = True -- ^ If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        })

loss :: RnnParams -> Tensor -> (Tensor, Tensor) -> Tensor
loss model initialState (input, target) =
    let (output, _) = forward model initialState input
    in mseLoss output target

forward :: RnnParams -> Tensor -> Tensor -> (Tensor,Tensor) -- ^ an output of (<seqLen,D*oDim>,(<D*numLayers,oDim>,<D*numLayers,cDim>)) so (output, (final hidden state, final cell state))
forward model initialState input = output
    where
        output = rnnLayers model Relu Nothing initialState input

-- randomize and initialize embedding with loaded params
initialize ::
  ModelSpec ->
  FilePath ->
  IO Model
initialize modelSpec embPath = do
  randomizedModel <- sample modelSpec
  loadedEmb <- loadParams (emb randomizedModel) embPath
  return Model {emb = loadedEmb , rnn = rnn randomizedModel}

-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "app/rnn/data/train.jsonl"

outputPath :: FilePath
outputPath = "app/rnn/data/review-texts.txt"

embeddingPath =  "app/rnn/models/sample_model_dim16_num6000.pt"

wordLstPath = "app/rnn/data/sample_wordlst.txt"

decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview]
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl
  in sequenceA $ map eitherDecode jsonList

rnnMain :: IO ()
rnnMain = do
  jsonl <- B.readFile amazonReviewPath
  let amazonReviews = decodeToAmazonReview jsonl
  let reviews = case amazonReviews of
                  Left err -> []
                  Right reviews -> reviews

  -- load word list (It's important to use the same list as whan creating embeddings)
  wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)

  -- load params (set　wordDim　and wordNum same as session5)
  let modelSpec = ModelSpec {
    wordDim = 16,
    wordNum = 6000
  }
  initModel <- initialize modelSpec embeddingPath

  print initModel
  print $ head reviews
  return ()
  where
    numEpochs = 300 :: Int

    modelPath :: String
    modelPath =  "app/rnn/models/sample_model.pt"

    numLayers = 2
    hDim = 6
    iDim = 1

    hyperParams :: RnnHypParams
    hyperParams = RnnHypParams {
        dev = Device CPU 0,
        bidirectional = False, -- ^ True if BiLSTM, False otherwise
        inputSize = iDim, -- ^ The number of expected features in the input x
        hiddenSize = hDim, -- ^ The number of features in the hidden state h
        numLayers = numLayers, -- ^ Number of recurrent layers
        hasBias = True -- ^ If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    }

    -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
    itr = 0 :: Int
    beta1 = 0.9 :: Float
    beta2 = 0.999 :: Float
    lr = 1e-1 :: Tensor
    dim = Dim 0 :: Dim