// scripts.js
document.addEventListener('DOMContentLoaded', function () {
    let dropArea = document.getElementById('drop-area');
    let loadingGif = document.getElementById('loading-gif');
    let statusDiv = document.getElementById('status-updates');

    // Hide loading GIF and status updates initially
    loadingGif.style.display = 'none';
    statusDiv.style.display = 'none';

    if (!dropArea) {
        console.error("Drop area not found!");
        return;
    }

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
    });

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let items = dt.items;

        let formData = new FormData();

        function traverseFileTree(item, path = "") {
            if (item.isFile) {
                item.file(function (file) {
                    let fullPath = path + file.name;
                    formData.append(fullPath, file);
                });
            } else if (item.isDirectory) {
                let dirReader = item.createReader();
                dirReader.readEntries(function (entries) {
                    for (let i = 0; i < entries.length; i++) {
                        traverseFileTree(entries[i], path + item.name + "/");
                    }
                });
            }
        }

        for (let i = 0; i < items.length; i++) {
            let item = items[i].webkitGetAsEntry();
            if (item) {
                traverseFileTree(item);
            }
        }

        // Clear status updates and show loading GIF
        statusDiv.innerHTML = ''; // Clear previous status updates
        dropArea.style.display = 'none';
        loadingGif.style.display = 'block';
        statusDiv.style.display = 'block';

        setTimeout(function () {
            fetch('/process', {  // Send POST to /process endpoint
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok ' + response.statusText);
                }
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');

                function read() {
                    return reader.read().then(({done, value}) => {
                        if (done) {
                            loadingGif.style.display = 'none';
                            return;
                        }
                        let chunk = decoder.decode(value);
                        if (chunk.includes('PROCESSING_COMPLETE')) {
                            let [header, ...rest] = chunk.split('PROCESSING_COMPLETE\n');
                            let treeAndJobId = rest.join('PROCESSING_COMPLETE\n');  // In case 'PROCESSING_COMPLETE' appears elsewhere
                            let [treeContent, jobIdLine] = treeAndJobId.split('\nJOB_ID:');
                            let jobId = jobIdLine.trim();
                            let tree = treeContent.trim();
                
                            document.body.innerHTML = `
                                <h1>Paper Ready</h1>
                                <h2>Directory Tree:</h2>
                                <pre>${tree}</pre>
                                <a href="/download?job_id=${jobId}" class="download-link">Download Restructured Files</a>
                                <a href="/reset?job_id=${jobId}" class="go-back-button">Go Back</a>
                            `;
                            loadingGif.style.display = 'none';
                            return;
                        } else {
                            statusDiv.innerHTML += chunk + '<br>';
                        }
                        return read();
                    });
                }                

                return read();
            })
            .catch(error => {
                console.error('Error during fetch:', error);
                loadingGif.style.display = 'none';
                statusDiv.innerHTML += 'An error occurred during processing.<br>';
                dropArea.style.display = 'block';
            });
        }, 1000);
    }
});
