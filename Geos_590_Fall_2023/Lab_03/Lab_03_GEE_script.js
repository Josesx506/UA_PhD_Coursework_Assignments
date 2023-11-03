// GEOS 590 - Fall Semester 2023
// lab 3
// Name: Joses
// Date: 11/02/2023
// Purpose of Code: Visualize change in NDVI values begore and after the 2020 Bay Area Fire in California
// Data Type: MODIS AQUA 16-day Vegetation Indices Global 250m
// Data Link: https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MYD13Q1


// 3. Import an image collection and extract NDVI, Near Infrared (NIR) and 
// Shortwave Infrared (SWIR) layers as individual variables.
var myd13q1 = ee.ImageCollection("MODIS/006/MYD13Q1");
print(myd13q1, 'MODIS Aqua 16-day 250m');

// Extract NDVI, NIR, and SWIR bands as variables
var ndvi = myd13q1.select('NDVI');
var nir = myd13q1.select('sur_refl_b02');
var swir = myd13q1.select('sur_refl_b07');


// 4. Import shapefile containing polygon of Bay Area Fire parameter
var bayarea_shp = ee.FeatureCollection("projects/gee-geos590-lab2/assets/Bay_Area_2020_Fire_shp");
var burn_area = bayarea_shp;


// 5. Mask image collection using Bay_Area_Parameter shapefile from assets folder.
// Create a function to reduce ImageCollection to the extent of
// the Bay Area burn area shapefile
// This function also applies a scaling factor (0.0001) to the values
var maskFunc = function(image){
  // Burn area is not specified as a function argument. Is it referring to the global variable?
  var masked = image.clip(burn_area).multiply(0.0001).rename('scaled');
  return image.addBands(masked);
};

// Run the mask function over the entire ImageCollection
// The ".map" function allows you to iterate over every image layer within the collection
var burn_area_ndvi = ndvi.map(maskFunc);
var burn_area_nir = nir.map(maskFunc);
var burn_area_swir = swir.map(maskFunc);


// 6. Subset image collection by date and create new variables for data corresponding with the 
// dates immediately before and after the Bay Area Fire.
// Set the start/end dates
var startDate = '2020-08-04';
var endDate = '2020-09-05'; // 2020-09-05|21

// Extract the pre & post fire ImageCollection Layers as individual image variables
// Pre-fire
var burn_area_ndvi_pre = burn_area_ndvi.filter(ee.Filter.date(startDate)).first().select('scaled').rename('Pre_NDVI');
var burn_area_nir_pre = burn_area_nir.filter(ee.Filter.date(startDate)).first().select('scaled').rename('NIR');
var burn_area_swir_pre = burn_area_swir.filter(ee.Filter.date(startDate)).first().select('scaled').rename('SWIR');
// Post-fire
var burn_area_ndvi_post = burn_area_ndvi.filter(ee.Filter.date(endDate)).first().select('scaled').rename('Post_NDVI');
var burn_area_nir_post = burn_area_nir.filter(ee.Filter.date(endDate)).first().select('scaled').rename('NIR');
var burn_area_swir_post = burn_area_swir.filter(ee.Filter.date(endDate)).first().select('scaled').rename('SWIR');

// Combine pre & post NDVI variables into a single image variable 
//  that will be used later to generate histogram
var ndvi_for_hist = burn_area_ndvi_pre.addBands(burn_area_ndvi_post.select('Post_NDVI'));

// Create a function to compute NDVI difference
var NDVIdiffFunc = function(image){
  var diff = image.expression('pre - post', 
    {'pre': image.select('Pre_NDVI'),
    'post': image.select('Post_NDVI')}).rename('NDVI_diff');
    return diff;
};

var ndvi_diff = NDVIdiffFunc(ndvi_for_hist);


// 7. Compute Normalized Burn Ratio (NBR) using NIR and SWIR data from before and after the Bay Area Fire.
// Combine the NIR and SWIR bands into pre and post image variables
var modisbands_pre = burn_area_nir_pre.addBands(burn_area_swir_pre.select('SWIR'));
var modisbands_post = burn_area_nir_post.addBands(burn_area_swir_post.select('SWIR'));

// Create NBR function
var NBRfunc = function(image){
  var nbr = image.expression('(nir - swir) / (nir + swir)', 
    {'nir': image.select('NIR'),
     'swir': image.select('SWIR')
    }).rename('NBR');
  return nbr;
};

var nbr_pre = NBRfunc(modisbands_pre);
var nbr_post = NBRfunc(modisbands_post);

// Compute NBR difference (dNBR) and multiply results by 1000 for later visualization
var dNBR = nbr_pre.subtract(nbr_post).multiply(1000).rename('dNBR');

// 8. Generating histogram and timeseries figures
// Create histogram plot showing distribution of pre-and-post fire NDVI
// Pre-define graphical customization options
var histOptions = {title: 'Pre & Post Bay Area Fire NDVI (MYD13Q1 v06)',
                   fontSize: 12,
                   hAxis: {title: 'NDVI', format: 'short'},
                   vAxis: {title: 'Pixel Count'},
                   series: {0: {color: 'green'},
                            1: {color: 'red'}}
};

// Make histogram using the prespecified options
var histogram = ui.Chart.image.histogram(ndvi_for_hist, burn_area, 30, 20)
                                        .setSeriesNames([startDate, endDate])
                                        .setOptions(histOptions)

//  Display the histogram within the console
print(histogram);


// 9. Create time series plot showing mean NDVI within the burn scar from 2016 and the present.
// Plot a time-series figure for the mean NDVI going back at laeast 5 years
var tsStart = '2015-08-20';
var tsStop = '2023-08-20';

//  Filter the NDVI ImageCollection using your selected dates
var ndvi_for_ts = burn_area_ndvi.filter(ee.Filter.date(tsStart, tsStop));

// Pre-define graphical customization options
var tsOptions = {title: 'Mean NDVI within Bay Area Fire burn scar // August 2015-2023',
                 fontSize: 12,
                 hAxis: {title: 'Date', format: 'MM-yyyy'},
                 vAxis: {title: 'Mean NDVI [ ]',viewWindow: {min: 0.25, max: 0.7}},
                 gridlines: {count: 8}
};

// Make timeseries with the set options argument
var ndviTS = ui.Chart.image.series(ndvi_for_ts.select('scaled'),
             bayarea_shp,
             ee.Reducer.mean(),
             100).setSeriesNames(['NDVI']).setOptions(tsOptions);

// Display the timeseries figure in the console window
print(ndviTS);


// 10. Exporting raster and tabular data to Google Drive
// Exporting the dNBR image as a Geotiff
Export.image.toDrive({
  image: dNBR,
  maxPixels: 1.0E13,
  description: 'Bay_Area_dNBR',
  scale: 500,
  region: burn_area,
  fileFormat: 'GeoTIFF'
});

// Export the daily mean NDVI timeseries values as a .csv file
Export.table.toDrive({
  collection: ndvi_for_ts,
  description: 'Daily_Mean_NDVI_2015_to_2023',
  fileFormat: 'CSV'
});


// 11. Visualizing data in the map window
// Set up NDVI visualization options
var ndviVis = {
  min: 0.0,
  max: 1.0,
  palette: ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
            '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
            '012E01', '011D01', '011301']
};

// Set up NDVI difference visualization options
var ndviDiffVis = {
  min: -0.5,
  max: 0.5,
  palette: ['3399FF', '66B2FF', '99CCFF', 'CCE5FF', 'FFFFFF',
            'FFCCCC', 'FF9999', 'FF6666', 'FF3333']
};

// Set up dNBR visualization options
var sld_intervals = '<RasterSymbolizer>' + 
                    '<ColorMap type="intervals" extended="false">' +
                    '<ColorMapEntry color="#ffffff" quantity="-500" label="-500"/>' + 
                    '<ColorMapEntry color="#7a8737" quantity="-250" label="-250"/>' + 
                    '<ColorMapEntry color="#acbe4d" quantity="-100" label="-100"/>' + 
                    '<ColorMapEntry color="#0ae042" quantity="100" label="100"/>' + 
                    '<ColorMapEntry color="#fff70b" quantity="270" label="270"/>' + 
                    '<ColorMapEntry color="#ffaf38" quantity="440" label="440"/>' + 
                    '<ColorMapEntry color="#ff641b" quantity="660" label="660"/>' + 
                    '<ColorMapEntry color="#a41fd6" quantity="2000" label="2000"/>' + 
                    '</ColorMap>' + '</RasterSymbolizer>';


// 12. Set default base map to “Terrain” and adjust the map center coordinates and zoom level
// Set basemap to "terrain"
Map.setOptions("TERRAIN");
// Set the map center and zoom level
Map.setCenter(-121.80, 37.3, 9);

// Add layers to map display
Map.addLayer(burn_area_ndvi_pre, ndviVis,'4 Aug NDVI');
Map.addLayer(burn_area_ndvi_post, ndviVis,'5 Sept NDVI');
Map.addLayer(ndvi_diff, ndviDiffVis,'Aug minus Sept NDVI');
Map.addLayer(dNBR.sldStyle(sld_intervals), {},'Burn Severity');

// Export the images. They were too large for export.
// Export.image.toDrive({
//   image: burn_area_ndvi_pre,
//   description: 'burn_area_ndvi_pre',
//   scale: 9,
//   region: burn_area
// });

// Export.image.toDrive({
//   image: burn_area_ndvi_post,
//   description: 'burn_area_ndvi_post',
//   scale: 9,
//   region: burn_area
// });

// Export.image.toDrive({
//   image: ndvi_diff,
//   description: 'ndvi_diff',
//   scale: 9,
//   region: burn_area
// });

// Export.image.toDrive({
//   image: dNBR.sldStyle(sld_intervals),
//   description: 'dNBR',
//   scale: 9,
//   region: burn_area
// });

// 14. Add Bay_Area_Parameter shapefile to map display window
var empty = ee.Image().byte();

//Paint the polygon edges
var burn_outline = empty.paint({
  featureCollection: burn_area,
  color: 1,
  width: 2
});




// ------------------------------------------------
// Add colorbar legends to the map display
// So much code for such a standard map feature :( 
// ------------------------------------------------

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NDVI colorbar
// // Set position of panel
// var ndviLegend = ui.Panel({
//   layout: ui.Panel.Layout.flow('horizontal'),
//   style: {
//     position: 'bottom-right',
//     padding: '30x 30px',
//     color: '000000'
//   }
// });
 
// // Create legend title
// var ndviLegendTitle = ui.Label({
//   value: 'NDVI [ ]',
//   style: {
//     fontWeight: 'bold',
//     fontSize: '16px',
//     margin: '0 0 0 0',
//     padding: '0'
//   }
// });
  
// // Add the title to the panel
// ndviLegend.add(ndviLegendTitle); 
  
// // Create the legend image
// var lon = ee.Image.pixelLonLat().select('longitude');
// var gradient = lon.multiply((ndviVis.max-ndviVis.min)/100.0).add(ndviVis.min);
// var legendImage = gradient.visualize(ndviVis);
  
// // Create text on top of legend
// var panel = ui.Panel({
//   widgets: [
//     ui.Label(ndviVis['min'])
//   ],
// });
  
// // ndviLegend.add(panel);
  
// // Create thumbnail from the image
// var thumbnail = ui.Thumbnail({
//   image: legendImage,
//   params: {bbox:'0,0,100,10', dimensions:'100x20'},  
//   style: {padding: '1px', position: 'bottom-center'},
// });
  
// // Add the thumbnail to the legend
// ndviLegend.add(thumbnail);
  
// // Create text on bottom of legend
// var panel = ui.Panel({
//   widgets: [
//     ui.Label(ndviVis['max'])
//   ],
// });

// ndviLegend.add(panel);
// Map.add(ndviLegend);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NDVI diff. colorbar
// Set position of panel
var diffLegend = ui.Panel({
  layout: ui.Panel.Layout.flow('horizontal'),
  style: {
    position: 'bottom-left',
    padding: '30x 30px',
    color: '000000'
  }
});
 
// Create legend title
var diffLegendTitle = ui.Label({
  value: 'NDVI diff.',
  style: {
    fontWeight: 'bold',
    fontSize: '16px',
    margin: '0 0 0 0',
    padding: '0'
  }
});
  
// Add the title to the panel
diffLegend.add(diffLegendTitle); 
  
// Create the legend image
var lon = ee.Image.pixelLonLat().select('longitude');
var gradient = lon.multiply((ndviDiffVis.max-ndviDiffVis.min)/100.0).add(ndviDiffVis.min);
var difflegendImage = gradient.visualize(ndviDiffVis);
  
// Create text on top of legend
var diffpanel = ui.Panel({
  widgets: [
    ui.Label(ndviDiffVis['min'])
  ],
});
  
diffLegend.add(diffpanel);
  
// Create thumbnail from the image
var thumbnail = ui.Thumbnail({
  image: difflegendImage,
  params: {bbox:'0,0,100,10', dimensions:'100x20'},  
  style: {padding: '1px', position: 'bottom-center'},
});
  
// Add the thumbnail to the legend
diffLegend.add(thumbnail);
  
// Create text on bottom of legend
var diffpanel = ui.Panel({
  widgets: [
    ui.Label(ndviDiffVis['max'])
  ],
});

diffLegend.add(diffpanel);
Map.add(diffLegend);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Burn severity (classified dNBR) colorbar
// // Set position of panel
// var dnbrlegend = ui.Panel({
//   style: {
//     position: 'bottom-center',
//     padding: '8px 15px'
// }});
 
// // Create legend title
// var dnbrlegendTitle = ui.Label({
//   value: 'Burn Severity (classified dNBR)',
//   style: {fontWeight: 'bold',
//     fontSize: '16px',
//     margin: '0 0 4px 0',
//     padding: '0'
// }});
 
// // Add the title to the panel
// dnbrlegend.add(dnbrlegendTitle);
 
// // Creates and styles 1 row of the legend.
// var makeRow = function(color, name) {
 
//       // Create the label/box icon.
//       var colorBox = ui.Label({
//         style: {
//           backgroundColor: '#' + color,
//           // Use padding to give the box height and width.
//           padding: '8px',
//           margin: '0 0 4px 0'
//       }});
 
//       // Create the label filled with the description text.
//       var description = ui.Label({
//         value: name,
//         style: {margin: '0 0 4px 6px'}
//       });
 
//       // return the panel
//       return ui.Panel({
//         widgets: [colorBox, description],
//         layout: ui.Panel.Layout.Flow('horizontal')
//       })};
 
// //  Palette with the colors
// var dnbrpalette =['7a8737', 'acbe4d', '0ae042', 'fff70b', 'ffaf38', 'ff641b', 'a41fd6', 'ffffff'];
 
// // name of the legend
// var dnbrclassnames = ['Enhanced Regrowth, High','Enhanced Regrowth, Low','Unburned', 'Low Severity',
// 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity', 'NA'];
 
// // Add color and and names
// for (var i = 0; i < 8; i++) {
//   dnbrlegend.add(makeRow(dnbrpalette[i], dnbrclassnames[i]));
// }  
 
// // add legend to map (alternatively you can also print the legend to the console)
// Map.add(dnbrlegend);