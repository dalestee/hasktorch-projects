{-# OPTIONS_GHC -Wno-identities #-}
module PCA (pca) where
import Numeric.LinearAlgebra (eigSH, toLists, fromLists, trustSym)
import Data.List (transpose)


-- z = (x - mean(x)) / std(x))
standardization :: [[Double]] -> [[Double]]
standardization xs = map (\x -> map (\(x', mean', std') -> (x' - mean') / std') (zip3 x means stds)) xs
  where
    means = map (\x -> sum x / fromIntegral (length x)) xs
    stds = map (\x -> sqrt (sum (map (\x' -> (x' - mean x) ^ (2 :: Integer)) x) / fromIntegral (length x))) xs

mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

covarianceMatrix :: [[Double]] -> [[Double]]
covarianceMatrix xs = map (\x -> map (covariance x) xs) xs
  where
    covariance x y = sum (zipWith (*) x y) / fromIntegral (length x)

pca :: [[Double]] -> [[Double]]
pca xs = map (\x -> map (dotProduct x) (transpose eigenvectorsDouble)) xs
  where
    eigenvectors = toLists . snd $ eigSH (trustSym . fromLists $ covarianceMatrix (standardization xs))
    eigenvectorsDouble = map (map realToFrac) eigenvectors
    dotProduct x y = sum (zipWith (*) x y)

-- | Example

main :: IO ()
main = do
  let xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  print $ pca xs


