/*
Copyright (c) 2015 The Polymer Project Authors. All rights reserved.
This code may only be used under the BSD style license found at http://polymer.github.io/LICENSE.txt
The complete set of authors may be found at http://polymer.github.io/AUTHORS.txt
The complete set of contributors may be found at http://polymer.github.io/CONTRIBUTORS.txt
Code distributed by Google as part of the polymer project is also
subject to an additional IP rights grant found at http://polymer.github.io/PATENTS.txt
*/

(function(document) {
    'use strict';

    // Grab a reference to our auto-binding template
    // and give it some initial binding values
    // Learn more about auto-binding templates at http://goo.gl/Dx1u2g
    var app = document.querySelector('#app');

    app.displayInstalledToast = function() {
        // Check to make sure caching is actually enabled—it won't be in the dev environment.
        if (!document.querySelector('platinum-sw-cache').disabled) {
            document.querySelector('#caching-complete').show();
        }
    };

    // Listen for template bound event to know when bindings
    // have resolved and content has been stamped to the page
    app.addEventListener('dom-change', function() {
        console.log('Our app is ready to rock!');
    });

    // Saves the table and data
    var hot;
    window.data = {
        header: '',
        values: ''
    };

    // Create a table
    document.addEventListener('json-loaded', function(e) {
        var container = document.getElementById('table');

        data.values = e.detail.values;
        data.header = e.detail.header;
        // If hot is undefined, create a hot table
        if (hot === undefined) {
            hot = new Handsontable(container, {
                data: data.values,
                height: 396,
                colHeaders: data.header,
                rowHeaders: true,
                stretchH: 'all',
                columnSorting: true,
                readOnly: true
            });

            hot.updateSettings({
                contextMenu: {
                    callback: function(key, options) {
                        if (key === 'about') {
                            setTimeout(function() {
                                // timeout is used to make sure the menu collapsed before alert is shown
                                alert("This is a context menu with default and custom options mixed");
                            }, 100);
                        }
                        if (key === 'inspect') {
                            var featureInspector = document.getElementById('bio-data-feature');
                            var selection = hot.getSelected();
                            var sortedData = hot.getData(0, 0, hot.countRows() - 1, hot.countCols() - 1); //a bit hacky to get the sorted data
                            featureInspector.extractFeatureData(selection, sortedData); // Call the featureInspector
                        }
                    },
                    items: {
                        "inspect": {
                            name: 'Inspect Feature',
                            disabled: function() {
                                // if select more than a row
                                var val = hot.getSelected(); 
                                return val[1] !== val[3]
                            }
                        },
                        "hsep1": "---------",
                        "remove_row": {
                            name: 'Remove this row, ok?'
                        },
                        "hsep2": "---------",
                        "about": {
                            name: 'About this menu'
                        }
                    }
                }
            })
        } else {
            hot.updateSettings({
                data: data.values,
                colHeaders: data.header
            })
            hot.render(); //otherwise just render new data
        }
    })

    document.addEventListener('json-not-found', function(e) {
        var container = document.getElementById('table');
        container.textContent = "Sry. We couldn't find your CSV file.";

    })

    // See https://github.com/Polymer/polymer/issues/1381
    window.addEventListener('WebComponentsReady', function() {
        // imports are loaded and elements have been registered
        var drawerPanel = document.querySelector('#paperDrawerPanel');
        drawerPanel.responsiveWidth = "1200px"
    });

    // Close drawer after menu item is selected if drawerPanel is narrow
    app.onDataRouteClick = function() {
        var drawerPanel = document.querySelector('#paperDrawerPanel');
        if (drawerPanel.narrow) {
            drawerPanel.closeDrawer();
        }
    };

    // Scroll page to top and expand header
    app.scrollPageToTop = function() {
        document.getElementById('mainContainer').scrollTop = 0;
    };

})(document);
