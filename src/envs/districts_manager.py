import os
import shapefile
from shapely.geometry import Polygon, Point, MultiPolygon, LinearRing, MultiLineString, MultiPoint, LineString
import geopandas as gpd

import utils

DISTRICTS_GEOM_NAME = 'DISTRITOS'
CENTROIDS = 'geoch_centroid.shp'
ORIGIN_X = 440165.384
ORIGIN_Y = 4474316.010

class DistrictsManager():
    def __init__(self, districts_geom_folder):
        self.folder = districts_geom_folder
        self.shapes = shapefile.Reader(os.path.join(districts_geom_folder, DISTRICTS_GEOM_NAME)).shapes()
        self.polygons = gpd.read_file(os.path.join(districts_geom_folder, DISTRICTS_GEOM_NAME + '.shp'))
        self.centroids = self._create_centroids(os.path.join(districts_geom_folder, CENTROIDS))
        self.districts_list = self.polygons.NOMBRE.tolist()

        self.districts_corrected = self._create_districts_corrected()
        self.centroids_corrected = self._create_centroids_corrected()

    def _create_centroids(self, centroids_file):
        if not os.path.isfile(centroids_file):
            points = self.polygons.copy()
            points.geometry = points['geometry'].centroid
            points.crs = self.polygons.crs
            points.to_file(centroids_file)
 
        centroids_shp = shapefile.Reader(centroids_file)

        nombres = self.polygons.NOMBRE.tolist()
        x_lists = []
        y_lists = []
        for centroid in centroids_shp.shapes():
            x = [i[0] for i in centroid.points[:]]
            y = [i[1] for i in centroid.points[:]]
            x_lists.append(x)
            y_lists.append(y)

        centroids = list(zip(nombres, x_lists, y_lists))

        return centroids

    def _create_districts_corrected(self):
        distritos_corrected = []
        for shape in self.shapes:
            x = [i[0] for i in shape.points[:]]
            y = [i[1] for i in shape.points[:]]
            new_x = []
            new_y = []
            for i in x:
                new_x.append(i - ORIGIN_X)
            for i in y:
                new_y.append(i - ORIGIN_Y)
            merged_list = tuple(zip(new_x, new_y))  
            distritos_corrected.append(Polygon(merged_list))

        return distritos_corrected

    def _create_centroids_corrected(self):
        centroides_corrected = []
        for centroide in self.centroids:
            x = centroide[1][0] - ORIGIN_X
            y = centroide[2][0] - ORIGIN_Y
            centroides_corrected.append(Point(x, y))

        return centroides_corrected

    def _get_district_polygon_by_name(self, name):
        index = self.districts_list.index(name)
        return self.districts_corrected[index]

    def obtain_route_cuts(self, origin, destination, districtName = None):
        names = self.districts_list if districtName == None else [districtName]
        line = LineString([Point(origin[0], origin[1]), Point(destination[0], destination[1])]) 

        result = {}
        for name in names:
            distrito = self._get_district_polygon_by_name(name)
            # To allow handle more than 2 intersection points between line and polygon
            distrito_lr = LinearRing(list(distrito.exterior.coords))
            intersection = distrito_lr.intersection(line)
            # Intersects district in ONE point
            if type(intersection) == Point:
                result[name]=[(intersection.x,intersection.y)]
            # Intersects district in TWO OR MORE point
            if type(intersection) == MultiPoint:
                result[name] = [(point.x, point.y) for point in intersection]

        return result

    # Calculate distances traversed across districts
    def get_segments_per_district(self, district_origin, origin, district_destination, destination):
        cuts = self.obtain_route_cuts(origin, destination)
        cuts[district_origin].insert(0, origin)
        cuts[district_destination].append(destination)

        distances = {k: utils.cartesian(*v) for (k,v) in cuts.items()}
        distances['Missing'] = utils.cartesian(origin,destination) - sum(distances.values())
        
        return distances