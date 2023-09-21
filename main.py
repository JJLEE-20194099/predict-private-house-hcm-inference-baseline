import os
import json
from fastapi import FastAPI
from fastapi import HTTPException, File, UploadFile
from dataclasses import astuple
from fastapi.responses import FileResponse
import requests
from tqdm import tqdm
from pydantic import BaseModel
import csv
import pandas as pd
from dataclasses import dataclass
import numpy as np
import lightgbm as lgb
import math
from typing import Union, List, Optional
import regex as re
from datetime import datetime

Num = Union[int, float]
class Distance:
    RADIUS = 6371
    MULTIPLE_TRANSFORM_COEFF = 10 / 1000
    DISTANCE_THRESHOLD = 10000


app = FastAPI()

@dataclass
class LocationConfig:
    lat: float
    lon: float
    distance: int

class MeanOfFacility:
    TOWNHALL = 'townhall'
    COMMUNITY_CENTER = 'community_centre'

@dataclass
class PlaceInformationConfig:
    park_df: pd.DataFrame
    road_df: pd.DataFrame
    commercial_df: pd.DataFrame

uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

dicchar = loaddicchar()
def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_cau_tieng_viet(sentence):

    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)
def covert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def preprocess_text(text):
    try:
        text = text.lower()
        text = covert_unicode(text)
        text = chuan_hoa_dau_cau_tieng_viet(text)
        return text
    except:
        return np.nan

def mapping_administrative_genre(administrative_genre):
    if administrative_genre == "quận":
        return 1
    if administrative_genre == "huyện":
        return 0
    return 0

async def findpublicfacilities(location_config: LocationConfig):

    lat, lon, distance = astuple(location_config)
    overpass_url = f"http://65.109.112.52/api/interpreter"

    overpass_query = f"""
      [out:json];
      (
      node["amenity"=""](around:{distance},{lat},{lon});
      way["amenity"=""](around:{distance},{lat},{lon});
      rel["amenity"=""](around:{distance},{lat},{lon});
      );
      out center;
      """

    response = requests.get(overpass_url,
                            params={'data': overpass_query}, timeout=60)

    data = response.json()

    return data['elements']

async def count_facilities(lat: float, lon: float, distance: int, response = None):
    if response is not None:
        res = response
    else:
        res = await findpublicfacilities(LocationConfig(
            lat = lat,
            lon = lon,
            distance = distance
        ))
    facility_dict = {}
    means_of_facility_obj = json.load(open('./app/files/json/means_of_facility.json', encoding='utf-8'))
    means_of_facility = means_of_facility_obj["means_of_facility_list"]
    for _fa in means_of_facility:
        facility_dict[_fa] = 0

    for place in res:
        try:
            if place['type'] == MeanOfFacility.TOWNHALL or place['type'] == MeanOfFacility.COMMUNITY_CENTER:
                facility_dict[f'{MeanOfFacility.TOWNHALL - MeanOfFacility.COMMUNITY_CENTER}'] = facility_dict[f'{MeanOfFacility.TOWNHALL - MeanOfFacility.COMMUNITY_CENTER}'] + 1
            if place['tags']['amenity'] in means_of_facility:
                facility_dict[place['tags']['amenity']
                            ] = facility_dict[place['tags']['amenity']] + 1

        except:
            pass

    return facility_dict


def distance_func(lat1: float, lon1: float, lat2: float, lon2: float):

    try:
        R = Distance.RADIUS
        dLat = (lat2-lat1) * math.pi / 180
        dLon = (lon2-lon1) * math.pi / 180
        lat1 = lat1 * math.pi / 180
        lat2 = lat2 * math.pi / 180
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.sin(dLon/2) * \
            math.sin(dLon/2) * math.cos(lat1) * math.cos(lat2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = R * c
        return d*1000
    except: # pylint: disable=bare-except
        print(f"Distance_Func error with: {(lat1, lon1, lat2, lon2)}")


def cal_distance(lat1_list: List[Num], lon1_list: List[Num], lat2_list: List[Num], lon2_list: List[Num]):
    try:
        all_res = []
        for i, lat2 in enumerate(lat2_list):
            lon2 = lon2_list[i]
            each_location_res = []
            for j, lat1 in enumerate(lat1_list):
                lon1 = lon1_list[j]
                each_location_res.append(distance_func(lat1, lon1, lat2, lon2))
            all_res.append(each_location_res)

        return all_res
    except: # pylint: disable=bare-except
        print(
            f"cal_distance error with: {(lat1_list, lon1_list, lat2_list, lon2_list)}")


def far_or_not(x: float, distance: float):
    try:
        if x > distance:
            return distance * Distance.MULTIPLE_TRANSFORM_COEFF
        return x / distance
    except: # pylint: disable=bare-except
        print(f"far_or_not with: {(x, distance)}")


def far_or_not_by_list(distance_input_arr, distance):
    try:
        all_res = []
        for instance in distance_input_arr:
            each_res = []
            for dis_candidate in instance:
                each_res.append(far_or_not(dis_candidate, distance))
            all_res.append(each_res)
        return all_res
    except: # pylint: disable=bare-except
        print(f"far_or_not_by_list with: {(distance_input_arr, distance)}")


def preprocess_distance(distance):
    if distance > Distance.MULTIPLE_TRANSFORM_COEFF:
        return np.nan
    return distance
def format_distance(distance, threshold=2000):
    if distance <= threshold:
        return distance / threshold
    return -1

def cal_distance_to_type_of_place(location_df:pd.DataFrame, place_information_config: PlaceInformationConfig):

    park_df, road_df, commercial_df = astuple(place_information_config)

    park_lat_list = park_df.lat.tolist()
    park_lon_list = park_df.lon.tolist()
    park_cols = [f'park_{i}' for i in range(len(park_df))]
    for i, col in enumerate(park_cols):
        location_df[col] = location_df.apply(lambda x: distance_func(
            x['lat'], x['lon'], park_lat_list[i], park_lon_list[i]), axis=1)

    road_lat_list = road_df.lat.tolist()
    road_lon_list = road_df.lon.tolist()
    road_cols = [f'road_{i}' for i in range(len(road_df))]
    for i, col in enumerate(road_cols):
        location_df[col] = location_df.apply(lambda x: distance_func(
            x['lat'], x['lon'], road_lat_list[i], road_lon_list[i]), axis=1)

    commercial_lat_list = commercial_df.lat.tolist()
    commercial_lon_list = commercial_df.lon.tolist()
    commercial_cols = [f'commercial_{i}' for i in range(len(commercial_df))]

    for col, commercial_lat, commercial_lon, in zip(commercial_cols, commercial_lat_list, commercial_lon_list):
        location_df[col] = location_df.apply(lambda x: distance_func(
            x['lat'], x['lon'], commercial_lat, commercial_lon), axis=1)

    return location_df




def calculate_distance_to_district_center(street_lat, street_lon, district_lat, district_lon):
    return distance_func(street_lat, street_lon, district_lat, district_lon)

trained_cols = ['num_of_floor',
 'park_2',
 'park_14',
 'fast_food',
 'road_3',
 'average_price',
 'population',
 'area_times',
 'road_21',
 'commercial_4',
 'park_10',
 'road_23',
 'administrative_genre',
 'park_12',
 'district_lat',
 'district_num',
 'park_7',
 'num_of_ward',
 'road_28',
 'commercial_0',
 'district_lon',
 'parking',
 'road_17',
 'commercial_2',
 'road_2',
 'road_16',
 'park_3',
 'commercial_6',
 'park_5',
 'road_19',
 'park_1',
 'road_5',
 'school',
 'townhall - community_centre',
 'land_area',
 'parking_entrance',
 'park_13',
 'road_27',
 'day',
 'road_7',
 'road_15',
 'park_4',
 'park_15',
 'commercial_1',
 'road_20',
 'road_14',
 'park_11',
 'road_26',
 'used_area',
 'juridical',
 'road_18',
 'bathroom',
 'road_22',
 'atm',
 'lat',
 'density',
 'park_8',
 'commercial_3',
 'street_num',
 'park_0',
 'hospital',
 'road_24',
 'place_of_worship',
 'road_0',
 'w',
 'bank',
 'commercial_5',
 'month',
 'restaurant',
 'road_12',
 'h',
 'lon',
 'bedroom',
 'marketplace',
 'alley_width',
 'is_nha_mt',
 'ward_num',
 'reg_area',
 'university',
 'cafe',
 'park_9',
 'acreage',
 'road_6',
 'road_11',
 'type_of_alley',
 'park_6',
 'road_10',
 'fuel',
 'road_1',
 'road_13',
 'road_8',
 'police',
 'year',
 'road_9',
 'road_25',
 'kindergarten',
 'road_4',
 'diff']
distance_cols = [c for c in trained_cols if 'commercial_' in c or 'park_' in c or 'road_' in c]

lgbm_regressor = lgb.Booster(model_file='./app/files/base_model/lgbm/lgbr_base.txt')
# prediced_price = np.exp(load_lgbm_regressor.predict(merge_df[trained_cols]))
# print(prediced_price, f"Min: {prediced_price * 0.8} - Max: {prediced_price * 1.2}")

district_dict = json.load(open('./app/files/json/dict/district.json', 'r', encoding="utf8"))
street_str_to_num_dict = json.load(open('./app/files/json/dict/street_str_to_num_dict.json', 'r', encoding="utf8"))
ward_str_to_num_dict = json.load(open('./app/files/json/dict/ward_str_to_num_dict.json', 'r', encoding="utf8"))
predicted_value_bias_by_district_dict = json.load(open('./app/files/json/dict/predicted_value_bias_by_district.json', 'r', encoding="utf8"))

population_df = pd.read_csv('./app/files/table/hcm_population.csv')
population_df['name'] = population_df['name'].str.strip().tolist()

park_df = pd.read_csv('./app/files/table/hcm_best_park.csv')
commercial_df = pd.read_csv('./app/files/table/hcm_best_commercial_center_place.csv')
road_df = pd.read_csv('./app/files/table/hcm_best_road_place.csv')

means_of_facility_obj = json.load(open('./app/files/json/means_of_facility.json', encoding='utf-8'))
means_of_facility = means_of_facility_obj["means_of_facility_list"]
distance = 2000

population_sum = population_df['population'].sum()


street_house_config = {
    "alley_width": [np.nan],
    "is_nha_mt": 1,
    # hợp đồng hợp lệ
    "juridical": 3,
    "type_of_alley": np.nan
}

alley_width_smaller_than_3_config = {
    "alley_width": [i / 10 for i in range(20, 30)],
    "is_nha_mt": 3,
    # hợp đồng hợp lệ
    "juridical": 3,
    "type_of_alley": 3
}

alley_width_3_and_6_config = {
    "alley_width": [i / 10 for i in range(30, 60)],
    "is_nha_mt": 3,
    # hợp đồng hợp lệ
    "juridical": 3,
    "type_of_alley": 2
}

alley_width_greater_than_10_config = {
    "alley_width": [i / 10 for i in range(60, 100)],
    "is_nha_mt": 3,
    # hợp đồng hợp lệ
    "juridical": 3,
    "type_of_alley": 1
}

type_of_config_list = [street_house_config, alley_width_greater_than_10_config, alley_width_3_and_6_config, alley_width_smaller_than_3_config]
initial_cols = [c for c in trained_cols if c not in ['juridical', 'alley_width', 'is_nha_mt']]


@app.get("//healthcheck")
def healthcheck():
    """Function checking price prediction module"""
    return {
        "data": "200"
    }

class GeolocationModel(BaseModel):
    latitude: float
    longitude: float


class RealEstateData(BaseModel):
    landSize: float
    city: str
    district: str
    ward: str
    prefixDistrict: str
    street: str
    country: str
    addressDetails: Optional[str]
    numberOfFloors: Optional[float]
    numberOfBathRooms: Optional[float]
    numberOfBedRooms: Optional[float]
    houseDirection: Optional[str]
    accessibility: Optional[str]
    frontWidth: Optional[float]
    geolocation: GeolocationModel
    extract_geolocation: GeolocationModel

@app.post("//predict-private-house-v1")
async def predict_price(body: RealEstateData):
    data = dict(body)

    for key in ["district", "ward", "street", "prefixDistrict"]:
        data[key] = preprocess_text(data[key])
    data["district_num"] = district_dict[data["district"]]
    data["ward_num"] = ward_str_to_num_dict[data["ward"]]
    data["street_num"] = street_str_to_num_dict[data["street"]]

    population_info = population_df[population_df['name'] == data['district']]
    data["administrative_genre"] = population_info['administrative_genre'].tolist()[0]
    data["acreage"] = population_info['acreage'].tolist()[0]
    data["population"] = population_info['population'].tolist()[0]
    data["density"] = population_info['density'].tolist()[0]
    data["district_lat"] = population_info['lat'].tolist()[0]
    data["district_lon"] = population_info['lon'].tolist()[0]
    data["average_price"] = population_info['average_price'].tolist()[0]
    data["num_of_ward"] = population_info['num_of_ward'].tolist()[0]

    facility_df = pd.DataFrame(columns = ['lat', 'lon'])
    facility_df['lat'] = [dict(data["geolocation"])["latitude"]]
    facility_df['lon'] = [dict(data["geolocation"])["longitude"]]

    num_of_facility = await count_facilities(dict(data["geolocation"])["latitude"], dict(data["geolocation"])["longitude"], distance)

    for facility in means_of_facility:
        data[facility] = num_of_facility[facility]
        facility_df[facility] = num_of_facility[facility]


    merge_df = cal_distance_to_type_of_place(facility_df, PlaceInformationConfig(
        park_df = park_df,
        road_df = road_df,
        commercial_df = commercial_df
    ))



    for c in distance_cols:
        merge_df[c] = merge_df[c].apply(format_distance)

    del data['geolocation']
    del data['extract_geolocation']
    for key in list(data.keys()):
        if data[key] == -1:
            merge_df[key] = [np.nan]
        else:
            merge_df[key] = [data[key]]
    merge_df['area_ratio'] = merge_df['landSize'] / (merge_df['acreage'] * 1000 * 1000) * 100
    merge_df['distance_point_and_center'] = merge_df.apply(lambda x: calculate_distance_to_district_center(x['lat'], x['lon'], x['district_lat'], x['district_lon']), axis=1)
    merge_df['distance_point_and_center'] = merge_df.apply(lambda x: format_distance(x['distance_point_and_center'], 5000), axis=1)
    merge_df['population'] = merge_df['population'] / population_sum * 100
    merge_df['type_of_alley'] = [None]

    now = datetime.now()
    month = now.month
    year = now.year
    day = now.day

    merge_df['day'] = [day]
    merge_df['month'] = [month]
    merge_df['year'] = [year]
    merge_df = merge_df.rename(columns = {'numberOfBathRooms': 'bathroom', 'numberOfBedRooms': 'bedroom', 'numberOfFloors': 'num_of_floor', 'landSize': 'land_area', 'frontWidth': 'w'})

    merge_df['h'] = merge_df['land_area'] / merge_df['w']
    merge_df['used_area'] = merge_df['land_area'] * merge_df['num_of_floor']
    merge_df['area_times'] = merge_df['num_of_floor']
    merge_df['diff'] = 0
    merge_df['reg_area'] = merge_df['land_area']

    merge_df['administrative_genre'] = merge_df['administrative_genre'].apply(mapping_administrative_genre)


    bias_level = predicted_value_bias_by_district_dict[str(merge_df['district_num'].tolist()[0])]

    stats_list = []
    for config in type_of_config_list:

        X = merge_df.copy()[initial_cols]
        for key in list(config.keys()):
            X[key] = [config[key]]
        alley_width_list = config['alley_width']
        s = 0
        for alley_width in alley_width_list:

            X["alley_width"] = [alley_width]
            X = X[trained_cols]
            predicted_price = np.exp(lgbm_regressor.predict(X))[0]
            # print()
            # print(X)
            # print(predicted_price)
            # print()
            s += predicted_price

        

        s = s / len(alley_width_list)

        stats = {
                "min": s - s * bias_level / 100,
                "max": s + s * bias_level / 100,
                "mean": s
            }
        stats_list.append(stats)

    return {
        "data": {
            "alleyHousePrice_1": stats_list[1],
            "alleyHousePrice_2": stats_list[2],
            "alleyHousePrice_3": stats_list[3],
            "streetHousePrice": stats_list[0]
        }
    }

