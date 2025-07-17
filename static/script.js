// Update file name in upload box when a file is selected
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    uploadBox.querySelector('span').innerText = fileInput.files[0].name;
  }
});

// Function to handle file upload and redirect to result page
function interceptFile() {
  const file = document.getElementById('fileInput').files[0];
  const mainContainer = document.getElementById('mainContainer');

  if (!file) {
    alert("Please upload a file first.");
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  // Show processing message
  mainContainer.innerHTML = `
    <h1>Processing your file... please wait ⏳</h1>
    <div class="loader"></div>
  `;

  // POST to backend
  fetch('http://localhost:5000/upload', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    console.log(data);
    console.log(data.result_url);
    // Redirect to result page
    window.location.href = data.result_url;
  })
  .catch(error => {
    console.error('Error:', error);
    mainContainer.innerHTML = `
      <h1 style="color: red;">An error occurred while processing your file.</h1>
      <p style="font-size: 16px; margin-top: 20px;">Please try again.</p>
    `;
  });
}
