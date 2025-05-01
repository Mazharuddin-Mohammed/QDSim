#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <Eigen/Dense>

// Simple mesh class for interpolation
class SimpleMesh {
public:
    SimpleMesh(const std::vector<Eigen::Vector2d>& nodes, const std::vector<std::array<int, 3>>& elements)
        : nodes(nodes), elements(elements) {}

    const std::vector<Eigen::Vector2d>& getNodes() const { return nodes; }
    const std::vector<std::array<int, 3>>& getElements() const { return elements; }

private:
    std::vector<Eigen::Vector2d> nodes;
    std::vector<std::array<int, 3>> elements;
};

// Simple interpolator class
class SimpleInterpolator {
public:
    SimpleInterpolator(const SimpleMesh& mesh) : mesh(mesh) {}

    double interpolate(double x, double y, const std::vector<double>& field) const {
        // Find the element containing the point
        int element_idx = findElement(x, y);
        if (element_idx < 0) {
            // Point is outside the mesh, return 0
            return 0.0;
        }

        // Get the element nodes
        const auto& elements = mesh.getElements();
        const auto& nodes = mesh.getNodes();
        const auto& element = elements[element_idx];

        // Compute barycentric coordinates
        std::vector<double> lambda(3);
        std::vector<Eigen::Vector2d> vertices = {
            nodes[element[0]], nodes[element[1]], nodes[element[2]]
        };
        computeBarycentricCoordinates(x, y, vertices, lambda);

        // Interpolate using barycentric coordinates
        return lambda[0] * field[element[0]] + lambda[1] * field[element[1]] + lambda[2] * field[element[2]];
    }

    std::tuple<double, double, double> interpolateWithGradient(double x, double y, const std::vector<double>& field) const {
        // Find the element containing the point
        int element_idx = findElement(x, y);
        if (element_idx < 0) {
            // Point is outside the mesh, return 0
            return std::make_tuple(0.0, 0.0, 0.0);
        }

        // Get the element nodes
        const auto& elements = mesh.getElements();
        const auto& nodes = mesh.getNodes();
        const auto& element = elements[element_idx];

        // Compute barycentric coordinates
        std::vector<double> lambda(3);
        std::vector<Eigen::Vector2d> vertices = {
            nodes[element[0]], nodes[element[1]], nodes[element[2]]
        };
        computeBarycentricCoordinates(x, y, vertices, lambda);

        // Interpolate using barycentric coordinates
        double value = lambda[0] * field[element[0]] + lambda[1] * field[element[1]] + lambda[2] * field[element[2]];

        // Compute the gradient
        double det = (vertices[1][1] - vertices[2][1]) * (vertices[0][0] - vertices[2][0]) +
                     (vertices[2][0] - vertices[1][0]) * (vertices[0][1] - vertices[2][1]);

        double grad_x = ((field[element[0]] * (vertices[1][1] - vertices[2][1]) +
                          field[element[1]] * (vertices[2][1] - vertices[0][1]) +
                          field[element[2]] * (vertices[0][1] - vertices[1][1])) / det);

        double grad_y = ((field[element[0]] * (vertices[2][0] - vertices[1][0]) +
                          field[element[1]] * (vertices[0][0] - vertices[2][0]) +
                          field[element[2]] * (vertices[1][0] - vertices[0][0])) / det);

        return std::make_tuple(value, grad_x, grad_y);
    }

    int findElement(double x, double y) const {
        const auto& elements = mesh.getElements();
        const auto& nodes = mesh.getNodes();

        for (size_t i = 0; i < elements.size(); ++i) {
            const auto& element = elements[i];
            std::vector<Eigen::Vector2d> vertices = {
                nodes[element[0]], nodes[element[1]], nodes[element[2]]
            };

            std::vector<double> lambda(3);
            if (computeBarycentricCoordinates(x, y, vertices, lambda)) {
                return i;
            }
        }

        return -1;
    }

private:
    const SimpleMesh& mesh;

    bool computeBarycentricCoordinates(double x, double y, const std::vector<Eigen::Vector2d>& vertices, std::vector<double>& lambda) const {
        // Compute barycentric coordinates
        double det = (vertices[1][1] - vertices[2][1]) * (vertices[0][0] - vertices[2][0]) +
                     (vertices[2][0] - vertices[1][0]) * (vertices[0][1] - vertices[2][1]);

        if (std::abs(det) < 1e-10) {
            return false;
        }

        lambda[0] = ((vertices[1][1] - vertices[2][1]) * (x - vertices[2][0]) +
                     (vertices[2][0] - vertices[1][0]) * (y - vertices[2][1])) / det;

        lambda[1] = ((vertices[2][1] - vertices[0][1]) * (x - vertices[2][0]) +
                     (vertices[0][0] - vertices[2][0]) * (y - vertices[2][1])) / det;

        lambda[2] = 1.0 - lambda[0] - lambda[1];

        // Check if the point is inside the triangle
        return lambda[0] >= -1e-10 && lambda[1] >= -1e-10 && lambda[2] >= -1e-10;
    }
};

// Forward declarations of Python types
// These will be defined later in the file

// Python wrapper for SimpleMesh
typedef struct {
    PyObject_HEAD
    SimpleMesh* mesh;
} PySimpleMesh;

// Python wrapper for SimpleInterpolator
typedef struct {
    PyObject_HEAD
    SimpleInterpolator* interpolator;
} PySimpleInterpolator;

// Deallocation function for PySimpleMesh
static void PySimpleMesh_dealloc(PySimpleMesh* self) {
    delete self->mesh;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Deallocation function for PySimpleInterpolator
static void PySimpleInterpolator_dealloc(PySimpleInterpolator* self) {
    delete self->interpolator;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Initialization function for PySimpleMesh
static int PySimpleMesh_init(PySimpleMesh* self, PyObject* args, PyObject* kwds) {
    PyObject* nodes_obj = nullptr;
    PyObject* elements_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &nodes_obj, &elements_obj)) {
        return -1;
    }

    // Convert nodes to std::vector<Eigen::Vector2d>
    std::vector<Eigen::Vector2d> nodes;
    if (PyList_Check(nodes_obj)) {
        Py_ssize_t size = PyList_Size(nodes_obj);
        nodes.reserve(size);

        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* node_obj = PyList_GetItem(nodes_obj, i);
            if (PyList_Check(node_obj) && PyList_Size(node_obj) == 2) {
                double x = PyFloat_AsDouble(PyList_GetItem(node_obj, 0));
                double y = PyFloat_AsDouble(PyList_GetItem(node_obj, 1));
                nodes.emplace_back(x, y);
            } else {
                PyErr_SetString(PyExc_TypeError, "Nodes must be a list of [x, y] coordinates");
                return -1;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Nodes must be a list");
        return -1;
    }

    // Convert elements to std::vector<std::array<int, 3>>
    std::vector<std::array<int, 3>> elements;
    if (PyList_Check(elements_obj)) {
        Py_ssize_t size = PyList_Size(elements_obj);
        elements.reserve(size);

        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* element_obj = PyList_GetItem(elements_obj, i);
            if (PyList_Check(element_obj) && PyList_Size(element_obj) == 3) {
                int n0 = PyLong_AsLong(PyList_GetItem(element_obj, 0));
                int n1 = PyLong_AsLong(PyList_GetItem(element_obj, 1));
                int n2 = PyLong_AsLong(PyList_GetItem(element_obj, 2));
                elements.push_back({n0, n1, n2});
            } else {
                PyErr_SetString(PyExc_TypeError, "Elements must be a list of [n0, n1, n2] indices");
                return -1;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Elements must be a list");
        return -1;
    }

    // Create the SimpleMesh
    self->mesh = new SimpleMesh(nodes, elements);

    return 0;
}

// Initialization function for PySimpleInterpolator
static int PySimpleInterpolator_init(PySimpleInterpolator* self, PyObject* args, PyObject* kwds) {
    PyObject* mesh_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O", &mesh_obj)) {
        return -1;
    }

    // Check if mesh_obj is a PySimpleMesh
    // We can't use PyObject_TypeCheck here because PySimpleMeshType is not defined yet
    // Instead, we'll check if the object has the expected attributes
    if (!PyObject_HasAttrString(mesh_obj, "mesh")) {
        // Try to extract the mesh pointer from the object
        SimpleMesh* mesh = (SimpleMesh*)PyCapsule_GetPointer(mesh_obj, "SimpleMesh");
        if (!mesh) {
            PyErr_SetString(PyExc_TypeError, "Mesh must be a SimpleMesh");
            return -1;
        }

        // Create the SimpleInterpolator
        self->interpolator = new SimpleInterpolator(*mesh);
    } else {
        // Get the SimpleMesh from the PySimpleMesh
        PySimpleMesh* py_mesh = (PySimpleMesh*)mesh_obj;

        // Create the SimpleInterpolator
        self->interpolator = new SimpleInterpolator(*py_mesh->mesh);
    }

    return 0;
}

// Method to interpolate a field at a point
static PyObject* PySimpleInterpolator_interpolate(PySimpleInterpolator* self, PyObject* args) {
    double x, y;
    PyObject* field_obj = nullptr;

    if (!PyArg_ParseTuple(args, "ddO", &x, &y, &field_obj)) {
        return nullptr;
    }

    // Convert field to std::vector<double>
    std::vector<double> field;
    if (PyList_Check(field_obj)) {
        Py_ssize_t size = PyList_Size(field_obj);
        field.reserve(size);

        for (Py_ssize_t i = 0; i < size; ++i) {
            double value = PyFloat_AsDouble(PyList_GetItem(field_obj, i));
            field.push_back(value);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Field must be a list");
        return nullptr;
    }

    // Interpolate the field
    double value = self->interpolator->interpolate(x, y, field);

    return PyFloat_FromDouble(value);
}

// Method to interpolate a field with gradient at a point
static PyObject* PySimpleInterpolator_interpolate_with_gradient(PySimpleInterpolator* self, PyObject* args) {
    double x, y;
    PyObject* field_obj = nullptr;

    if (!PyArg_ParseTuple(args, "ddO", &x, &y, &field_obj)) {
        return nullptr;
    }

    // Convert field to std::vector<double>
    std::vector<double> field;
    if (PyList_Check(field_obj)) {
        Py_ssize_t size = PyList_Size(field_obj);
        field.reserve(size);

        for (Py_ssize_t i = 0; i < size; ++i) {
            double value = PyFloat_AsDouble(PyList_GetItem(field_obj, i));
            field.push_back(value);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Field must be a list");
        return nullptr;
    }

    // Interpolate the field with gradient
    auto [value, grad_x, grad_y] = self->interpolator->interpolateWithGradient(x, y, field);

    // Create a tuple with the results
    PyObject* result = PyTuple_New(3);
    PyTuple_SetItem(result, 0, PyFloat_FromDouble(value));
    PyTuple_SetItem(result, 1, PyFloat_FromDouble(grad_x));
    PyTuple_SetItem(result, 2, PyFloat_FromDouble(grad_y));

    return result;
}

// Method to find the element containing a point
static PyObject* PySimpleInterpolator_find_element(PySimpleInterpolator* self, PyObject* args) {
    double x, y;

    if (!PyArg_ParseTuple(args, "dd", &x, &y)) {
        return nullptr;
    }

    // Find the element
    int element_idx = self->interpolator->findElement(x, y);

    return PyLong_FromLong(element_idx);
}

// Method table for PySimpleInterpolator
static PyMethodDef PySimpleInterpolator_methods[] = {
    {"interpolate", (PyCFunction)PySimpleInterpolator_interpolate, METH_VARARGS, "Interpolate a field at a point"},
    {"interpolate_with_gradient", (PyCFunction)PySimpleInterpolator_interpolate_with_gradient, METH_VARARGS, "Interpolate a field with gradient at a point"},
    {"find_element", (PyCFunction)PySimpleInterpolator_find_element, METH_VARARGS, "Find the element containing a point"},
    {nullptr}
};

// Type object for PySimpleMesh
static PyTypeObject PySimpleMeshType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "fe_interpolator_ext.SimpleMesh",  // tp_name
    sizeof(PySimpleMesh),              // tp_basicsize
    0,                                 // tp_itemsize
    (destructor)PySimpleMesh_dealloc,  // tp_dealloc
    0,                                 // tp_print
    0,                                 // tp_getattr
    0,                                 // tp_setattr
    0,                                 // tp_compare
    0,                                 // tp_repr
    0,                                 // tp_as_number
    0,                                 // tp_as_sequence
    0,                                 // tp_as_mapping
    0,                                 // tp_hash
    0,                                 // tp_call
    0,                                 // tp_str
    0,                                 // tp_getattro
    0,                                 // tp_setattro
    0,                                 // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "SimpleMesh objects",              // tp_doc
    0,                                 // tp_traverse
    0,                                 // tp_clear
    0,                                 // tp_richcompare
    0,                                 // tp_weaklistoffset
    0,                                 // tp_iter
    0,                                 // tp_iternext
    0,                                 // tp_methods
    0,                                 // tp_members
    0,                                 // tp_getset
    0,                                 // tp_base
    0,                                 // tp_dict
    0,                                 // tp_descr_get
    0,                                 // tp_descr_set
    0,                                 // tp_dictoffset
    (initproc)PySimpleMesh_init,       // tp_init
    0,                                 // tp_alloc
    PyType_GenericNew,                 // tp_new
};

// Type object for PySimpleInterpolator
static PyTypeObject PySimpleInterpolatorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "fe_interpolator_ext.SimpleInterpolator", // tp_name
    sizeof(PySimpleInterpolator),      // tp_basicsize
    0,                                 // tp_itemsize
    (destructor)PySimpleInterpolator_dealloc, // tp_dealloc
    0,                                 // tp_print
    0,                                 // tp_getattr
    0,                                 // tp_setattr
    0,                                 // tp_compare
    0,                                 // tp_repr
    0,                                 // tp_as_number
    0,                                 // tp_as_sequence
    0,                                 // tp_as_mapping
    0,                                 // tp_hash
    0,                                 // tp_call
    0,                                 // tp_str
    0,                                 // tp_getattro
    0,                                 // tp_setattro
    0,                                 // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "SimpleInterpolator objects",      // tp_doc
    0,                                 // tp_traverse
    0,                                 // tp_clear
    0,                                 // tp_richcompare
    0,                                 // tp_weaklistoffset
    0,                                 // tp_iter
    0,                                 // tp_iternext
    PySimpleInterpolator_methods,      // tp_methods
    0,                                 // tp_members
    0,                                 // tp_getset
    0,                                 // tp_base
    0,                                 // tp_dict
    0,                                 // tp_descr_get
    0,                                 // tp_descr_set
    0,                                 // tp_dictoffset
    (initproc)PySimpleInterpolator_init, // tp_init
    0,                                 // tp_alloc
    PyType_GenericNew,                 // tp_new
};

// Module definition
static PyModuleDef fe_interpolator_ext_module = {
    PyModuleDef_HEAD_INIT,
    "fe_interpolator_ext",
    "Extension module for finite element interpolation",
    -1,
    nullptr
};

// Module initialization function
PyMODINIT_FUNC PyInit_fe_interpolator_ext(void) {
    // Initialize NumPy
    import_array();

    // Initialize the module
    PyObject* m = PyModule_Create(&fe_interpolator_ext_module);
    if (m == nullptr) {
        return nullptr;
    }

    // Initialize the types
    if (PyType_Ready(&PySimpleMeshType) < 0 || PyType_Ready(&PySimpleInterpolatorType) < 0) {
        return nullptr;
    }

    // Add the types to the module
    Py_INCREF(&PySimpleMeshType);
    PyModule_AddObject(m, "SimpleMesh", (PyObject*)&PySimpleMeshType);

    Py_INCREF(&PySimpleInterpolatorType);
    PyModule_AddObject(m, "SimpleInterpolator", (PyObject*)&PySimpleInterpolatorType);

    return m;
}
