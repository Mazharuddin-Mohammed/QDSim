#!/usr/bin/env python3
"""
Minimal Exploration - Just check what's available
"""

def main():
    print("Minimal QDSim Exploration")
    print("="*40)
    
    try:
        import qdsim_cpp
        print("✅ qdsim_cpp imported")
        
        # Get all available items
        all_items = [x for x in dir(qdsim_cpp) if not x.startswith('_')]
        print(f"Total items: {len(all_items)}")
        
        # Show all items
        for i, item in enumerate(all_items, 1):
            print(f"{i:2d}. {item}")
        
        # Try to create just a mesh
        print(f"\nTesting Mesh creation...")
        mesh = qdsim_cpp.Mesh(50e-9, 50e-9, 8, 8, 1)
        print(f"✅ Mesh created: {mesh.get_num_nodes()} nodes")
        
        # Check mesh methods
        mesh_methods = [m for m in dir(mesh) if not m.startswith('_')]
        print(f"Mesh methods: {mesh_methods}")
        
        print(f"\n✅ Basic functionality working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
