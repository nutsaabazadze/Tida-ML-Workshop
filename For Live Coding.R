land_train <- readRDS('data/manhattan_Train.rds')
land_test <- readRDS('data/manhattan_Test.rds')

View(land_train)

valueFormula <- TotalValue ~ FireService + 
  ZoneDist1 + ZoneDist2 + Class + LandUse + 
  OwnerType + LotArea + BldgArea + ComArea + 
  ResArea + OfficeArea + RetailArea + 
  GarageArea + FactryArea + NumBldgs + 
  NumFloors + UnitsRes + UnitsTotal + 
  LotFront + LotDepth + BldgFront + 
  BldgDepth + LotType + Landmark + BuiltFAR +
  Built + HistoricDistrict - 1

value1 <- lm(valueFormula, data=land_train)
summary(value1)

str(land_train)


