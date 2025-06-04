
#!/usr/bin/env python3
"""Simple Cython functionality test"""

def test_materials():
    """Test materials module"""
    try:
        from qdsim_cython.core import materials
        
        # Test Material creation
        mat = materials.Material()
        mat.m_e = 0.041
        mat.epsilon_r = 13.9
        
        print(f"✅ Material: m_e={mat.m_e}, ε_r={mat.epsilon_r}")
        
        # Test MaterialDatabase
        db = materials.MaterialDatabase()
        print("✅ MaterialDatabase created")
        
        return True
        
    except Exception as e:
        print(f"❌ Materials test failed: {e}")
        return False

def test_mesh():
    """Test mesh module"""
    try:
        from qdsim_cython.core import mesh
        print("✅ Mesh module accessible")
        return True
        
    except Exception as e:
        print(f"❌ Mesh test failed: {e}")
        return False

def test_physics():
    """Test physics module"""
    try:
        from qdsim_cython.core import physics
        print("✅ Physics module accessible")
        return True
        
    except Exception as e:
        print(f"❌ Physics test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Cython Modules...")
    
    materials_ok = test_materials()
    mesh_ok = test_mesh()
    physics_ok = test_physics()
    
    if all([materials_ok, mesh_ok, physics_ok]):
        print("🎉 Cython modules working!")
    else:
        print("❌ Some Cython modules failed")
