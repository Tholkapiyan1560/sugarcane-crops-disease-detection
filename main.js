// Function to generate dynamic region input blocks based on the region count
function generateRegionBlocks() {
    const regionCount = parseInt(document.getElementById('region_count').value) || 1;
    const container = document.getElementById('region_container');
    container.innerHTML = ''; // Clear previous blocks

    for (let i = 1; i <= regionCount; i++) {
        let block = document.createElement('div');
        block.className = 'region-block';
        block.innerHTML = `
            <h4>Region ${i}</h4>
            <div class="form-group mb-3">
                <label for="region_${i}_start">Start Image Number:</label>
                <input type="number" name="region_${i}_start" id="region_${i}_start" class="form-control" required>
            </div>
            <div class="form-group mb-3">
                <label for="region_${i}_end">End Image Number:</label>
                <input type="number" name="region_${i}_end" id="region_${i}_end" class="form-control" required>
            </div>
            <div class="form-group mb-3">
                <label for="region_${i}_horizontal">Number of Horizontal Splits:</label>
                <input type="number" name="region_${i}_horizontal" id="region_${i}_horizontal" class="form-control" required>
            </div>
            <div class="form-group mb-3">
                <label for="region_${i}_vertical_preview">Vertical Preview? (yes/no):</label>
                <input type="text" name="region_${i}_vertical_preview" id="region_${i}_vertical_preview" class="form-control" placeholder="no" required>
            </div>
            <div class="form-group mb-3">
                <label for="region_${i}_num_vsplits">Number of Vertical Splits (for preview):</label>
                <input type="text" name="region_${i}_num_vsplits" id="region_${i}_num_vsplits" class="form-control" placeholder="e.g., 3">
            </div>
            <div class="form-group mb-3">
                <label for="region_${i}_vertical_split">Vertical Split Focus? (yes/no):</label>
                <input type="text" name="region_${i}_vertical_split" id="region_${i}_vertical_split" class="form-control" placeholder="no" required>
            </div>
            <div class="form-group mb-3">
                <label for="region_${i}_num_vertical_parts">Number of Vertical Parts (if splitting):</label>
                <input type="text" name="region_${i}_num_vertical_parts" id="region_${i}_num_vertical_parts" class="form-control" placeholder="e.g., 3">
            </div>
            <div class="form-group mb-3">
                <label for="region_${i}_selected_parts">Parts to Keep (e.g., 1-3 or 2):</label>
                <input type="text" name="region_${i}_selected_parts" id="region_${i}_selected_parts" class="form-control" placeholder="e.g., 1-3">
            </div>`;
        if (i > 1) {
            let orderField = document.createElement('div');
            orderField.className = 'form-group mb-3';
            orderField.innerHTML = `
                <label for="region_${i}_order">Region ${i} Order relative to Region 1 (L/R):</label>
                <input type="text" name="region_${i}_order" id="region_${i}_order" class="form-control" placeholder="L or R">
            `;
            block.appendChild(orderField);
        }
        container.appendChild(block);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    generateRegionBlocks();
    document.getElementById('region_count').addEventListener('change', generateRegionBlocks);
});
