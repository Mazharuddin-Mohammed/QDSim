#!/bin/bash
# Script to generate documentation for QDSim

# Create directories if they don't exist
mkdir -p docs/html
mkdir -p docs/pdf
mkdir -p docs/images

# Generate Doxygen documentation
echo "Generating Doxygen documentation..."
doxygen Doxyfile

# Generate PDF from Markdown files
echo "Generating PDF documentation..."
pandoc docs/user_guide/index.md -o docs/pdf/user_guide.pdf --toc --toc-depth=3 --highlight-style=tango --variable geometry:margin=1in
pandoc docs/theory/index.md -o docs/pdf/theory.pdf --toc --toc-depth=3 --highlight-style=tango --variable geometry:margin=1in

# Copy example scripts to docs/examples
echo "Copying example scripts..."
mkdir -p docs/examples
cp examples/*.py docs/examples/

# Generate HTML from Markdown files
echo "Generating HTML documentation..."
pandoc docs/user_guide/index.md -o docs/html/user_guide.html --toc --toc-depth=3 --highlight-style=tango --standalone --metadata title="QDSim User Guide"
pandoc docs/theory/index.md -o docs/html/theory.html --toc --toc-depth=3 --highlight-style=tango --standalone --metadata title="QDSim Theory Documentation"

# Create index.html
echo "Creating index.html..."
cat > docs/html/index.html << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QDSim Documentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #333;
        }
        a {
            color: #0066cc;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .card {
            flex: 1;
            min-width: 300px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .card ul {
            padding-left: 20px;
        }
        .footer {
            margin-top: 50px;
            border-top: 1px solid #eee;
            padding-top: 20px;
            text-align: center;
            color: #666;
        }
    </style>
</head>
<body>
    <h1>QDSim Documentation</h1>
    <p>Welcome to the QDSim documentation. QDSim is a 2D quantum dot simulator that solves the Schrödinger equation for quantum dots in semiconductor heterostructures.</p>
    
    <div class="container">
        <div class="card">
            <h2>User Guide</h2>
            <p>The user guide provides comprehensive documentation for QDSim, including installation instructions, usage examples, and best practices.</p>
            <ul>
                <li><a href="user_guide.html">HTML Version</a></li>
                <li><a href="../pdf/user_guide.pdf">PDF Version</a></li>
            </ul>
        </div>
        
        <div class="card">
            <h2>Theory Documentation</h2>
            <p>The theory documentation provides a detailed explanation of the theoretical foundations and numerical methods used in QDSim.</p>
            <ul>
                <li><a href="theory.html">HTML Version</a></li>
                <li><a href="../pdf/theory.pdf">PDF Version</a></li>
            </ul>
        </div>
        
        <div class="card">
            <h2>API Reference</h2>
            <p>The API reference provides detailed documentation for all classes and functions in QDSim.</p>
            <ul>
                <li><a href="api/index.html">API Reference</a></li>
            </ul>
        </div>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>Examples</h2>
            <p>The examples demonstrate how to use QDSim for various applications.</p>
            <ul>
                <li><a href="../examples/documentation_demo.py">Documentation Demo</a></li>
                <li><a href="../examples/error_handling_demo.py">Error Handling Demo</a></li>
                <li><a href="../examples/physical_accuracy_demo.py">Physical Accuracy Demo</a></li>
                <li><a href="../examples/chromium_qd_pn_diode.py">Chromium QD in P-N Diode</a></li>
            </ul>
        </div>
    </div>
    
    <div class="footer">
        <p>QDSim Documentation - Generated on $(date)</p>
        <p>Author: Dr. Mazharuddin Mohammed</p>
    </div>
</body>
</html>
EOF

echo "Documentation generation complete!"
echo "You can view the documentation by opening docs/html/index.html in your web browser."
