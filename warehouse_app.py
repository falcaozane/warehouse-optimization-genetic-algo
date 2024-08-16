import streamlit as st
from space_optimization import Warehouse, Product, space_allocation_genetic_algorithm

def main():
    st.title("Warehouse Space Allocation Genetic Algorithm")
    st.write("This app optimizes the space allocation in a warehouse using a genetic algorithm.")

    # Sidebar for inputs
    with st.sidebar:
        st.header("Warehouse Parameters")
        warehouse_length = st.number_input("Warehouse Length", 10)
        warehouse_width = st.number_input("Warehouse Width", 10)
        warehouse_height = st.number_input("Warehouse Height", 5)

        st.header("Define Products")
        products = []
        for i in range(1, 10):
            st.write(f"Product {i}")
            height = st.slider(f"Height {i}", 1, 5, 1)
            length = st.slider(f"Length {i}", 1, 6, 3)
            width = st.slider(f"Width {i}", 1, 5, 2)
            fragile = st.checkbox(f"Fragile {i}")
            demand = st.slider(f"Demand {i}", 50, 500, 250)
            stock = st.slider(f"Stock {i}", 0, 500, 100)
            product = Product(height=height, length=length, width=width, fragile=fragile, demand=demand, id=i+100, stock=stock)
            products.append(product)

        st.header("Genetic Algorithm Parameters")
        population_size = st.slider("Population Size", 10, 100, 20)
        generations = st.slider("Generations", 100, 5000, 4)

    # Create warehouse instance
    warehouse = Warehouse(warehouse_length, warehouse_width, warehouse_height)
    warehouse.add_entrance((0, 0))
    warehouse.block_area(initial=(0, 7, 0), dimensions=(3, 3, 5))
    warehouse.make_path([0, 1], [9, 1])
    warehouse.make_path([9, 1], [9, 3])
    warehouse.make_path([9, 3], [1, 3])
    warehouse.make_path([1, 3], [1, 5])
    warehouse.make_path([1, 5], [9, 5])
    warehouse.make_path([9, 5], [9, 7])
    warehouse.make_path([9, 7], [3, 7])
    warehouse.make_path([3, 7], [3, 9])
    warehouse.make_path([3, 9], [9, 9])

    st.subheader("Warehouse Layout")
    st.write("The warehouse layout is displayed below:")
    plot_html = warehouse.display_warehouse()
    st.markdown(plot_html, unsafe_allow_html=True)

    if st.button("Run Genetic Algorithm"):
        best_solution = space_allocation_genetic_algorithm(warehouse, products, population_size, generations)
        if not best_solution:
            st.error("Cannot find the best solution due to lack of space in the warehouse.")
        else:
            st.success("Best solution found:")
            warehouse.add_products(best_solution)
            plot_html = warehouse.display_warehouse()
            st.markdown(plot_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
