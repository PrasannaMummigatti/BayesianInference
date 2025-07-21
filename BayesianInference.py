import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import tempfile
import PIL.Image

st.set_page_config(layout="centered", page_title="Bayesian Coin Toss", page_icon="🪙")
st.markdown(
    """
    <script>
        window.scrollTo(0, 0);
    </script>
    """,
    unsafe_allow_html=True
)

st.title("🪙 Bayesian Inference: Estimate Coin Bias from 100 Tosses")

st.markdown("""
**Instructions:**  
- Enter exactly 100 comma-separated toss outcomes (0 = tail, 1 = head) Note end the input with a comma.  
- You can generate this data in Excel with: `=BINOM.INV(1, 0.7, RAND())`  *(replace 0.7 with your desired bias)*
- This is prepopulated with 100 tosses data with a bias of 0.7. Click "Submit Data" to compute the posterior distribution.
""")

input_text = st.text_area(
    "🔢 Enter 100 comma-separated coin toss outcomes:",
    height=70,
    value="0,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,0,1,1,"
)

if st.button("Submit Data"):
    data = []
    if input_text:
        try:
            cleaned = input_text.replace("\n", "").strip().split(",")
            data = [int(x.strip()) for x in cleaned if x.strip() != '']

            if len(data) != 100:
                st.error(f"❌ You entered {len(data)} values. Please enter exactly 100.")
                st.stop()
            elif not all(x in [0, 1] for x in data):
                st.error("❌ Only 0 and 1 values are allowed.")
                st.stop()
            else:
                st.success("✅ Valid input received. Computing posterior...")

                # Step 1: Create grid and uniform prior
                p_grid = np.linspace(0, 1, 200)
                prior = np.ones_like(p_grid)

                # Step 2: Sequentially update the posterior
                posteriors = []
                n_heads, n_tails = 0, 0
                for i in range(100):
                    if data[i] == 1:
                        n_heads += 1
                    else:
                        n_tails += 1
                    likelihood = p_grid**n_heads * (1 - p_grid)**n_tails
                    posterior_unnorm = likelihood * prior
                    posterior = posterior_unnorm / np.sum(posterior_unnorm)
                    posteriors.append(posterior)

                # Step 3: Animation setup
                fig, ax = plt.subplots(figsize=(8, 4))
                # Draw vertical line at MAP estimate (m
                # aximum of last posterior)
                map_idx = np.argmax(posteriors[-1])
                map_p = p_grid[map_idx]
                ax.axvline(map_p, color='red', linestyle='--', label=f'MAP: {map_p:.2f}')
                ax.legend()
                line, = ax.plot([], [], lw=2)
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, np.max(posteriors[-1]) * 1.1)
                ax.set_xlabel("Bias (p)")
                ax.set_ylabel("Probability Density")
                ax.set_title("Posterior Distribution of Coin Bias")
                text = ax.text(0.7, np.max(posteriors[-1]) * 0.9, "", fontsize=12)

                def init():
                    line.set_data([], [])
                    text.set_text("")
                    return line, text

                def animate(i):
                    y = posteriors[i]
                    line.set_data(p_grid, y)
                    text.set_text(f"Toss {i+1}")
                    return line, text

                ani = animation.FuncAnimation(
                    fig, animate, frames=100, init_func=init,
                    blit=True, interval=100, repeat=False
                )


                # Step 4: Save animation to a temporary GIF file
                #with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmpfile:
                #    ani.save(tmpfile.name, writer=PillowWriter(fps=10))
                #    st.image(tmpfile.name, caption="Posterior Animation", use_container_width =True)
                #    st.stop()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmpfile:
                    ani.save(tmpfile.name, writer=PillowWriter(fps=10))  # Save GIF

                    # Reopen and save with loop=0 using Pillow to disable looping
                    gif = PIL.Image.open(tmpfile.name)
                    frames = []
                    try:
                        while True:
                            frames.append(gif.copy())
                            gif.seek(len(frames))  # Move to next frame
                    except EOFError:
                        pass  # End of sequence

                    frames[0].save(tmpfile.name, save_all=True, append_images=frames[1:], loop=0)
                    st.image(tmpfile.name, caption="Posterior Animation", use_container_width=True)

                    # Static chart of final posterior
                    fig_static, ax_static = plt.subplots(figsize=(8, 4))
                    ax_static.plot(p_grid, posteriors[-1], lw=2, label="Posterior")
                    ax_static.axvline(map_p, color='red', linestyle='--', label=f'MAP: {map_p:.2f}')
                    ax_static.set_xlim(0, 1)
                    ax_static.set_xlabel("Bias (p)")
                    ax_static.set_ylabel("Probability Density")
                    ax_static.set_title("Final Posterior Distribution")
                    ax_static.legend()
                    st.pyplot(fig_static)

                    st.stop()
        except Exception as e:
            st.error(f"❌ Error processing input: {e}")
