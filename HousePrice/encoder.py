# ENCODING:
    # One-Hot selection is determined by the median response value associated with each class
    # of each predictor. Note that some features have 2+ classes for which median is higher than
    # most. In these instances, the highest median response value is used.

# TAGS:
#  'MSZoning' 'Street' 'Alley' 'LotShape' 'LandContour' 'Utilities'
#  'LotConfig' 'LandSlope' 'Neighborhood' 'Condition1' 'Condition2'
#  'BldgType' 'HouseStyle' 'RoofStyle' 'RoofMatl' 'Exterior1st'
#  'Exterior2nd' 'MasVnrType' 'ExterQual' 'ExterCond' 'Foundation'
#  'BsmtQual' 'BsmtCond' 'BsmtExposure' 'BsmtFinType1' 'BsmtFinType2'
#  'Heating' 'HeatingQC' 'CentralAir' 'Electrical' 'KitchenQual'
#  'Functional' 'FireplaceQu' 'GarageType' 'GarageFinish' 'GarageQual'
#  'GarageCond' 'PavedDrive' 'PoolQC' 'Fence' 'MiscFeature' 'SaleType'
#  'SaleCondition'

import pandas as pd
import numpy as np

def encode(train,test):

    # Feature Class = 2
    #------------------

    train['enc_Street'] = pd.get_dummies(train.Street, drop_first=True)
    test['enc_Street'] = pd.get_dummies(test.Street, drop_first=True)

    train['enc_Alley'] = pd.get_dummies(train.Alley, drop_first=True)
    test['enc_Alley'] = pd.get_dummies(test.Alley, drop_first=True)

    train['enc_CentralAir'] = pd.get_dummies(train.CentralAir, drop_first=True)
    test['enc_CentralAIr'] = pd.get_dummies(test.CentralAir, drop_first=True)

    # Feature Class > 2
    #------------------

    def enc(feature): return 1 if feature == 'AllPub' else 0
    train['enc_Utilities'] = train.Utilities.apply(enc)
    test['enc_Utilities'] = test.Utilities.apply(enc)

    def enc(feature): return 1 if feature == 'RL' else 0
    train['enc_MSZoning'] = train.MSZoning.apply(enc)
    test['enc_MSZoning'] = test.MSZoning.apply(enc)

    def enc(feature): return 1 if feature == 'Reg' else 0
    train['enc_LotShape'] = train.LotShape.apply(enc)
    test['enc_LotShape'] = test.LotShape.apply(enc)

    def enc(feature): return 1 if feature == 'Lvl' else 0
    train['enc_LandContour'] = train.LandContour.apply(enc)
    test['enc_LandContour'] = test.LandContour.apply(enc)

    def enc(feature): return 1 if feature == 'Inside' else 0
    train['enc_LotConfig'] = train.LotConfig.apply(enc)
    test['enc_LotConfig'] = test.LotConfig.apply(enc)

    def enc(feature): return 1 if feature == 'Gtl' else 0
    train['enc_LandSlope'] = train.LandSlope.apply(enc)
    test['enc_LandSlope'] = test.LandSlope.apply(enc)

    def enc(feature): return 1 if feature == 'Norm' else 0
    train['enc_Condition1'] = train.Condition1.apply(enc)
    test['enc_Condition1'] = test.Condition1.apply(enc)
    train['enc_Condition2'] = train.Condition2.apply(enc)
    test['enc_Condition2'] = test.Condition2.apply(enc)

    def enc(feature): return 1 if feature == '1Fam' else 0
    train['enc_BldgType'] = train.BldgType.apply(enc)
    test['enc_BldgType'] = test.BldgType.apply(enc)

    def enc(feature): return 1 if feature == 'Gable' else 0
    train['enc_RoofStyle'] = train.RoofStyle.apply(enc)
    test['enc_RoofStyle'] = test.RoofStyle.apply(enc)

    def enc(feature): return 1 if feature == 'CompShg' else 0
    train['enc_RoofMatl'] = train.RoofMatl.apply(enc)
    test['enc_RoofMatl'] = test.RoofMatl.apply(enc)

    def enc(feature): return 1 if feature == 'VinylSd' else 0
    train['enc_extr1st'] = train.Exterior1st.apply(enc)
    test['enc_extr1st'] = test.Exterior1st.apply(enc)
    train['enc_extr2nd'] = train.Exterior2nd.apply(enc)
    test['enc_extr2nd'] = test.Exterior2nd.apply(enc)

    def enc(feature): return 1 if feature == 'TA' else 0
    train['enc_extrCond'] = train.ExterCond.apply(enc)
    test['enc_extrCond'] = test.ExterCond.apply(enc)

    def enc(feature): return 1 if feature == 'TA' else 0
    train['enc_BsmtCond'] = train.BsmtCond.apply(enc)
    test['enc_BsmtCond'] = test.BsmtCond.apply(enc)

    def enc(feature): return 1 if feature == 'Ex' else 0
    train['enc_BsmtQual'] = train.BsmtQual.apply(enc)
    test['enc_BsmtQual'] = test.BsmtQual.apply(enc)

    def enc(feature): return 1 if feature == 'No' else 0
    train['enc_BsmtExposure'] = train.BsmtExposure.apply(enc)
    test['enc_BsmtExposure'] = test.BsmtExposure.apply(enc)

    def enc(feature): return 1 if feature == 'GasA' else 0
    train['enc_Heating'] = train.Heating.apply(enc)
    test['enc_Heating'] = test.Heating.apply(enc)

    def enc(feature): return 1 if feature == 'Ex' else 0
    train['enc_HeatingQC'] = train.HeatingQC.apply(enc)
    test['enc_HeatingQC'] = test.HeatingQC.apply(enc)

    def enc(feature): return 1 if feature == 'Unf' else 0
    train['enc_BsmtFinType1'] = train.BsmtFinType1.apply(enc)
    test['enc_BsmtFinType1'] = test.BsmtFinType1.apply(enc)
    train['enc_BsmtFinType2'] = train.BsmtFinType2.apply(enc)
    test['enc_BsmtFinType2'] = test.BsmtFinType2.apply(enc)

    def enc(feature): return 1 if feature == 'SBrkr' else 0
    train['enc_Electrical'] = train.Electrical.apply(enc)
    test['enc_Electrical'] = test.Electrical.apply(enc)

    def enc(feature): return 1 if feature == 'Gd' else 0
    train['enc_FireplaceQu'] = train.FireplaceQu.apply(enc)
    test['enc_FireplaceQu'] = test.FireplaceQu.apply(enc)

    def enc(feature): return 1 if feature == 'Ex' else 0
    train['enc_KitchenQual'] = train.KitchenQual.apply(enc)
    test['enc_KitchenQual'] = test.KitchenQual.apply(enc)

    def enc(feature): return 1 if feature == 'Typ' else 0
    train['enc_Functional'] = train.Functional.apply(enc)
    test['enc_Functional'] = test.Functional.apply(enc)

    def enc(feature): return 1 if feature == 'NridgHt' else 0
    train['enc_Neighborhood'] = train.Neighborhood.apply(enc)
    test['enc_Neighborhood'] = test.Neighborhood.apply(enc)

    def enc(feature): return 1 if feature == 'BuiltIn' else 0
    train['enc_GarageType'] = train.GarageType.apply(enc)
    test['enc_GarageType'] = test.GarageType.apply(enc)

    def enc(feature): return 1 if feature == 'Fin' else 0
    train['enc_GarageFinish'] = train.GarageFinish.apply(enc)
    test['enc_GarageFinish'] = test.GarageFinish.apply(enc)

    def enc(feature): return 1 if feature == 'Gd' else 0
    train['enc_GarageQual'] = train.GarageQual.apply(enc)
    test['enc_GarageQual'] = test.GarageQual.apply(enc)

    def enc(feature): return 1 if feature == 'TA' else 0
    train['enc_GarageCond'] = train.GarageCond.apply(enc)
    test['enc_GarageCond'] = test.GarageCond.apply(enc)

    def enc(feature): return 1 if feature == 'Y' else 0
    train['enc_PavedDrive'] = train.PavedDrive.apply(enc)
    test['enc_PavedDrive'] = test.PavedDrive.apply(enc)

    def enc(feature): return 1 if feature == 'Ex' else 0
    train['enc_PoolQC'] = train.PoolQC.apply(enc)
    test['enc_PoolQC'] = test.PoolQC.apply(enc)

    def enc(feature): return 1 if feature == 'GdPrv' else 0
    train['enc_Fence'] = train.Fence.apply(enc)
    test['enc_Fence'] = test.Fence.apply(enc)

    def enc(feature): return 1 if feature == 'TenC' else 0
    train['enc_MiscFeature'] = train.MiscFeature.apply(enc)
    test['enc_MiscFeature'] = test.MiscFeature.apply(enc)

    def enc(feature): return 1 if feature == 'Con' else 0
    train['enc_SaleType'] = train.SaleType.apply(enc)
    test['enc_SaleType'] = test.SaleType.apply(enc)

    def enc(feature): return 1 if feature == 'Partial' else 0
    train['enc_SaleCondition'] = train.SaleCondition.apply(enc)
    test['enc_SaleCondition'] = test.SaleCondition.apply(enc)

    def enc(feature): return 1 if feature == '1Story' else 0
    train['enc_HouseStyle'] = train.HouseStyle.apply(enc)
    test['enc_HouseStyle'] = test.HouseStyle.apply(enc)
    # 2Story

    def enc(feature): return 1 if feature == 'None' else 0
    train['enc_MasVnr'] = train.MasVnrType.apply(enc)
    test['enc_MasVnr'] = test.MasVnrType.apply(enc)
    # BrkFace

    def enc(feature): return 1 if feature == 'TA' else 0
    train['enc_ExterQual'] = train.ExterQual.apply(enc)
    test['enc_ExterQual'] = test.ExterQual.apply(enc)
    # GD

    def enc(feature): return 1 if feature == 'PConc' else 0
    train['enc_Foundation'] = train.Foundation.apply(enc)
    test['enc_Foundation'] = test.Foundation.apply(enc)
    # CBlock

    return train, test