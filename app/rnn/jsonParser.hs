{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}


module Main (main) where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import GHC.Generics
import Torch.NN (Parameter, Parameterized(..), Randomizable(..))
import Torch.Serialize (loadParams)
import Torch.TensorFactories (randnIO')
import Torch.Autograd (makeIndependent)

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


data Model = Model {
  emb :: Embedding
  -- TODO: add RNN
  -- rnn :: RNN
} deriving (Show, Generic, Parameterized)


instance
  Randomizable
    ModelSpec
    Model
  where
    sample ModelSpec {..} = 
        Model
        <$> Embedding <$> (makeIndependent =<< randnIO' [wordDim, wordNum])
        -- TODO: add RNN initilization
        -- <*> sample ...

-- randomize and initialize embedding with loaded params
initialize ::
  ModelSpec ->
  FilePath ->
  IO Model
initialize modelSpec embPath = do
  randomizedModel <- sample modelSpec
  loadedEmb <- loadParams (emb randomizedModel) embPath
  return Model {emb = loadedEmb {-, rnn = rnn randomizedModel -}}

-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "app/rnn/data/train.jsonl"

outputPath :: FilePath
outputPath = "app/rnn/data/review-texts.txt"

embeddingPath =  "app/rnn/data/sample_embedding.params"

wordLstPath = "app/rnn/data/sample_wordlst.txt"

decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview] 
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl
  in sequenceA $ map eitherDecode jsonList

main :: IO ()
main = do
  jsonl <- B.readFile amazonReviewPath
  let amazonReviews = decodeToAmazonReview jsonl
  let reviews = case amazonReviews of
                  Left err -> []
                  Right reviews -> reviews

  -- load word list (It's important to use the same list as whan creating embeddings)
  -- wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)

  -- load params (set　wordDim　and wordNum same as session5)
  let modelSpec = ModelSpec {
    wordDim = 9, 
    wordNum = 9
  }
  -- initModel <- initialize modelSpec embeddingPath

  -- print initModel
  print $ head reviews
  return ()