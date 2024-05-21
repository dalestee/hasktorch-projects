module MatrixOp (magnitude, dotProduct) where

magnitude :: [Float] -> Float
magnitude vec = sqrt $ sum $ map (^2) vec

dotProduct :: [Float] -> [Float] -> Float
dotProduct vec1 vec2 = sum $ zipWith (*) vec1 vec2